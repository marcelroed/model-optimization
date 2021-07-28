import tensorflow as tf
from tensorflow.keras import backend, Sequential
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.util import nest
from tensorflow.python.keras.engine import functional

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

layers = tf.keras.layers
keras = tf.keras


def _can_broadcast(shape1, shape2):
  """Whether two shapes can fit together"""
  for i, j in zip(reversed(shape1), reversed(shape2)):
    if i != j and i != 1 and j != 1:
      return False
  return True


class SimplificationContext:
  """Holds information passed on to the next node in the computation graph
  Use cases:
    1. Keeping track of the pruning that happened to the previous layer, so following layers can remove parts of the input dimension.
       Preserve a mask for the output layers that were pruned from the previous layer.
    2. When a layer prunes an output channel, it has biases that can be propagated through to the biases of the next layer.
       In this case we need to preserve the bias term from the previous layer.

  Also needs to be generic (not only work for convolutional layers, so that it can be applied also to inbetween layers.
  """
  def __init__(self, old_shape, did_prune=False, new_shape=None, bias=None):
    self.old_shape = old_shape
    self.new_shape = new_shape if new_shape is not None else old_shape
    self.did_prune = self.old_shape == self.new_shape
    self.bias = bias
    if bias is not None:
      assert _can_broadcast(self.bias.shape, self.old_shape), "Bias must be broadcastable with the output shape."


# Will hold functions of the form (context, layer) -> (context, layer)
_SIMPLIFICATION_IMPLEMENTATIONS = {}


def register_pruning_implementation(*layer_types, override=False):
  """Decorator to register implementations for filter pruning"""
  for layer_type in layer_types:
    if layer_type in _SIMPLIFICATION_IMPLEMENTATIONS.keys() and not override:
      raise ValueError(f'Layer type {layer_type} already defined with function \'{_SIMPLIFICATION_IMPLEMENTATIONS[layer_type].__name__}\'')

  def do_registration(fun):
    for layer_type in layer_types:
      _SIMPLIFICATION_IMPLEMENTATIONS[layer_type] = fun
    return fun
  return do_registration


@register_pruning_implementation
def _clone_layer(layer, context):
  """Basic 'do-nothing' clone implementation"""
  return layer.__class__.from_config(layer.get_config()), context


def _delegate_clone(layer, context):
  layer_class = layer.__class__
  if layer_class not in _SIMPLIFICATION_IMPLEMENTATIONS.keys():
    print(f'WARNING: The layer class {layer_class} has no simplification implementation, and will likely result in suboptimal structural pruning.')
  clone_function = _SIMPLIFICATION_IMPLEMENTATIONS[layer.__class__]
  return clone_function(layer, context)


def _simplify_conv(conv_layer: keras.layers.Conv2D):
  """Remove filters with completely zeroed weights in convolutional layer

  Raises:
      SimplificationFailedException: TODO: When?
  """

  kernel, bias = conv_layer.weights  # (w, h, in_channels, out_channels), (out_channels,)

  # Aggregate kernel weights on all axes but output filter
  filter_present = tf.reduce_max(kernel, axis=[0, 1, 2])  # -> (out_channels)

  present_filter_mask = tf.greater(tf.abs(filter_present), 1e-20)

  # TODO(marcelroed): Propagate the bias elements using the next layers weights, then remove it from the current layer

  new_output_dim = tf.reduce_sum(tf.cast(present_filter_mask, tf.int32))

  # Construct a new layer with the
  new_layer_config = conv_layer.get_config().copy()
  new_layer_config['filters'] = new_output_dim
  new_layer = conv_layer.__class__.from_config(new_layer_config)
  new_layer.build(input_shape=conv_layer.input_shape)

  new_kernel = tf.boolean_mask(kernel, present_filter_mask, axis=3)  # Kernel has fewer out_channels
  new_bias = tf.boolean_mask(bias,     present_filter_mask, axis=0)  # Bias has fewer out_channels

  new_layer.set_weights([new_kernel, new_bias])

  return new_layer, new_output_dim, present_filter_mask


def _conv_transform_weights_by_context(kernel, context: SimplificationContext):
  """Use the layout from the context to remove part of the kernel"""
  kernel_input_trimmed = tf.boolean_mask(kernel, context.prune_layout, axis=2)  # -> (w, h, new_in_channels, out_channels)
  return kernel_input_trimmed


def _make_filter_mask(kernel):
  """Reduce the kernel (w, h, in_channels, out_channels) -> (out_channels,) boolean mask
  Also return the number of present out_channels"""
  kernel_reduced_to_filters = tf.reduce_max(kernel, axis=[0, 1, 2])
  filter_mask_present = tf.greater(tf.abs(kernel_reduced_to_filters), 1e-20)
  new_out_channels = tf.reduce_sum(tf.cast(filter_mask_present, tf.int32))
  return filter_mask_present, new_out_channels


def _simplified_conv_layer(conv_layer, new_output_dim, new_input_shape):
  new_layer_config = conv_layer.get_config().copy()
  new_layer_config['filters'] = new_output_dim  # Set output channels
  new_layer = conv_layer.__class__.from_config(new_layer_config)
  new_layer.build(input_shape=new_input_shape)
  return new_layer


def _make_kernel_bias(kernel, bias, filters_present):
  """Makes the kernel and bias the right shape for inserting into the restructured model.
  Notes that this removes bias terms from the layer!
  Assumes that the kernel already has the input_shape resized.
  """
  new_kernel = tf.boolean_mask(kernel, filters_present, axis=3)
  new_bias = tf.boolean_mask(bias, filters_present, axis=0)
  return new_kernel, new_bias


@register_pruning_implementation(layers.Conv2D)
def _simplify_conv(conv_layer: layers.Conv2D, context: SimplificationContext):
  # Context should always come from an image, possibly with reduced channels.
  # Needs to also deal with reductions on itself.

  # Get existing weights and some information about them
  kernel, bias = conv_layer.get_weights()  # (w, h, in_channels, out_channels), (out_channels,)
  kernel_trimmed = _conv_transform_weights_by_context(kernel, context)
  filters_present, new_output_dim = _make_filter_mask(kernel_trimmed)

  # Save the old output shape and create a new layer with the correct config
  old_shape = conv_layer.output_shape
  new_conv_layer = _simplified_conv_layer(conv_layer, new_output_dim, new_input_shape=context.new_shape)

  # Reshape weights and insert them into the newly built model
  new_kernel, new_bias = _make_kernel_bias(kernel_trimmed, bias, filters_present)
  new_conv_layer.set_weights(new_kernel, new_bias)

  # Transform the bias using the current layer's activation if it's set
  if context.bias is not None:
    # TODO(marcelroed): Handle bias terms being passed forward
    pass

  # Set up context for next layer
  context = SimplificationContext(
    old_shape=old_shape,
    new_shape=new_conv_layer.output_shape,
  )

  return new_conv_layer, context


def _strip_pruning_with_simplification(model):
  """Strip pruning wrappers from the model, checking if output dimensions can be reduced.
  Propagate changes to later layers.

  Returns:
      A keras model with pruning wrappers removed.
  """

  # TODO(marcelroed): Test if this works for an arbitrary DAG (functional model)

  def _strip_pruning_wrapper(layer):
    if isinstance(layer, tf.keras.Model):
      # A keras model with prunable layers
      return keras.models.clone_model(
        layer, input_tensors=None, clone_function=_strip_pruning_wrapper)
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      # The _batch_input_shape attribute in the first layer makes a Sequential
      # model to be built. This makes sure that when we remove the wrapper from
      # the first layer the model's built state preserves.
      if not hasattr(layer.layer, '_batch_input_shape') and hasattr(
              layer, '_batch_input_shape'):
        layer.layer._batch_input_shape = layer._batch_input_shape
      return layer.layer
    return layer

  return keras.models.clone_model(
    model, input_tensors=None, clone_function=_strip_pruning_wrapper)


def clone_with_simplify(model, input_tensors=None, clone_function=_delegate_clone):
  """Clone a Functional or Sequential `Model` instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Note that
  `clone_model` will not preserve the uniqueness of shared objects within the
  model (e.g. a single variable attached to two distinct layers will be
  restored as two separate variables).

  Args:
      model: Instance of `Model`
          (could be a Functional model or a Sequential model).
      input_tensors: optional list of input tensors or InputLayer objects
          to build the model upon. If not provided,
          new `Input` objects will be created.
      clone_function: Callable to be used to clone each layer in the target
          model (except `InputLayer` instances). It takes as argument the layer
          instance to be cloned, and returns the corresponding layer instance to
          be used in the model copy. If unspecified, this callable defaults to
          the following serialization/deserialization function:
          `lambda layer: layer.__class__.from_config(layer.get_config())`.
          By passing a custom callable, you can customize your copy of the
          model, e.g. by wrapping certain layers of interest (you might want to
          replace all `LSTM` instances with equivalent
          `Bidirectional(LSTM(...))` instances, for example).

  Returns:
    An instance of `Model` reproducing the behavior
    of the original model, on top of new inputs tensors,
    using newly instantiated weights. The cloned model may behave
    differently from the original model if a custom `clone_function`
    modifies the layer.

  Example:

  ```python
  # Create a test Sequential model.
  model = keras.Sequential([
      keras.Input(shape=(728,)),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(1, activation='sigmoid'),
  ])
  # Create a copy of the test model (with freshly initialized weights).
  new_model = clone_model(model)
  ```

  Note that subclassed models cannot be cloned, since their internal
  layer structure is not known. To achieve equivalent functionality
  as `clone_model` in the case of a subclassed model, simply make sure
  that the model class implements `get_config()`
  (and optionally `from_config()`), and call:

  ```python
  new_model = model.__class__.from_config(model.get_config())
  ```
  """
  with generic_utils.DisableSharedObjectScope():
    # if clone_function is None:
    #   clone_function = _clone_layer

    if isinstance(model, keras.Sequential):
      return _clone_sequential_model(
          model, input_tensors=input_tensors, layer_fn=clone_function)
    else:
      return _clone_functional_model(
          model, input_tensors=input_tensors, layer_fn=clone_function)


def _clone_sequential_model(model, input_tensors=None, layer_fn=_delegate_clone):
  """Clone a `Sequential` model instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Args:
      model: Instance of `Sequential`.
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.
      layer_fn: callable to be applied on non-input layers in the model. By
          default it clones the layer. Another example is to preserve the layer
          to share the weights. This is required when we create a per-replica
          copy of the model with distribution strategy; we want the weights to
          be shared but still feed inputs separately so we create new input
          layers.

  Returns:
      An instance of `Sequential` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value or `layer_fn`
      argument value.
  """
  if not isinstance(model, keras.Sequential):
    raise ValueError('Expected `model` argument '
                     'to be a `Sequential` model instance, '
                     'but got:', model)

  if not callable(layer_fn):
    raise ValueError('Expected `layer_fn` argument to be a callable.')

  layers = []  # Layers needed to compute the model's outputs.
  layer_map = {}
  # Ensure that all layers are cloned. The model's layers
  # property will exclude the initial InputLayer (if it exists) in the model,
  # resulting in a different Sequential model structure.
  for layer in model._flatten_layers(include_self=False, recursive=False):
    if isinstance(layer, InputLayer) and input_tensors is not None:
      # If input tensors are provided, the original model's InputLayer is
      # overwritten with a different InputLayer.
      continue
    cloned_layer = (
      _clone_layer(layer)
      if isinstance(layer, InputLayer) else layer_fn(layer))
    layers.append(cloned_layer)
    layer_map[layer] = cloned_layer
  layers, ancillary_layers = _remove_ancillary_layers(model, layer_map, layers)

  if input_tensors is None:
    cloned_model = Sequential(layers=layers, name=model.name)
  elif len(generic_utils.to_list(input_tensors)) != 1:
    raise ValueError('To clone a `Sequential` model, we expect '
                     ' at most one tensor '
                     'as part of `input_tensors`.')
  else:
    # Overwrite the original model's input layer.
    if isinstance(input_tensors, tuple):
      input_tensors = list(input_tensors)
    x = generic_utils.to_list(input_tensors)[0]
    if backend.is_keras_tensor(x):
      origin_layer = x._keras_history.layer
      if isinstance(origin_layer, InputLayer):
        cloned_model = keras.Sequential(
          layers=[origin_layer] + layers, name=model.name)
      else:
        raise ValueError('Cannot clone a `Sequential` model on top '
                         'of a tensor that comes from a Keras layer '
                         'other than an `InputLayer`. '
                         'Use the functional API instead.')
    else:
      input_tensor = keras.Input(tensor=x, name='input_wrapper_for_' + str(x.name))
      input_layer = input_tensor._keras_history.layer
      cloned_model = keras.Sequential(layers=[input_layer] + layers, name=model.name)

  if not ancillary_layers:
    return cloned_model

  tensor_map = {}  # Maps tensors from `model` to those in `cloned_model`.
  for depth, cloned_nodes in cloned_model._nodes_by_depth.items():
    nodes = model._nodes_by_depth[depth]
    # This should be safe in a Sequential model. In an arbitrary network, you
    # need to sort using the outbound layer of the node as a key.
    for cloned_node, node in zip(cloned_nodes, nodes):
      if isinstance(cloned_node.output_tensors, list):
        for j, output_tensor in enumerate(cloned_node.output_tensors):
          tensor_map[node.output_tensors[j]] = output_tensor
      else:
        tensor_map[node.output_tensors] = cloned_node.output_tensors
  # Ancillary nodes have negative depth.
  new_nodes = _make_new_nodes(
    {
      depth: nodes
      for depth, nodes in model._nodes_by_depth.items()
      if depth < 0
    }, layer_fn, layer_map, tensor_map)
  _insert_ancillary_layers(cloned_model, ancillary_layers, model.metrics_names,
                           new_nodes)
  return cloned_model


def _clone_functional_model(model, input_tensors=None, layer_fn=_delegate_clone):
  """Clone a functional `Model` instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Input layers are always cloned.

  Args:
      model: Instance of `Model`.
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.
      layer_fn: callable to be applied on non-input layers in the model. By
          default it clones the layer. Another example is to preserve the layer
          to share the weights. This is required when we create a per-replica
          copy of the model with distribution strategy; we want the weights to
          be shared but still feed inputs separately so we create new input
          layers.

  Returns:
      An instance of `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value or `layer_fn`
      argument value.
  """
  if not isinstance(model, keras.Model):
    raise ValueError('Expected `model` argument '
                     'to be a `Model` instance, got ', model)
  if isinstance(model, keras.Sequential):
    raise ValueError('Expected `model` argument '
                     'to be a functional `Model` instance, '
                     'got a `Sequential` instance instead:', model)
  if not model._is_graph_network:
    raise ValueError('Expected `model` argument '
                     'to be a functional `Model` instance, '
                     'but got a subclass model instead.')

  new_input_layers = {}  # Cache for created layers.
  if input_tensors is not None:
    # Make sure that all input tensors come from a Keras layer.
    input_tensors = nest.flatten(input_tensors)
    for i, input_tensor in enumerate(input_tensors):
      original_input_layer = model._input_layers[i]

      # Cache input layer. Create a new layer if the tensor is originally not
      # from a Keras layer.
      if not backend.is_keras_tensor(input_tensor):
        name = original_input_layer.name
        input_tensor = keras.Input(tensor=input_tensor,
                             name='input_wrapper_for_' + name)
        newly_created_input_layer = input_tensor._keras_history.layer
        new_input_layers[original_input_layer] = newly_created_input_layer
      else:
        new_input_layers[original_input_layer] = original_input_layer

  if not callable(layer_fn):
    raise ValueError('Expected `layer_fn` argument to be a callable.')

  model_configs, created_layers = _clone_layers_and_model_config(
    model, new_input_layers, layer_fn)
  # Reconstruct model from the config, using the cloned layers.
  input_tensors, output_tensors, created_layers = (
    functional.reconstruct_from_config(model_configs,
                                       created_layers=created_layers))
  metrics_names = model.metrics_names
  model = keras.Model(input_tensors, output_tensors, name=model.name)
  # Layers not directly tied to outputs of the Model, such as loss layers
  # created in `add_loss` and `add_metric`.
  ancillary_layers = [
    layer for layer in created_layers.values() if layer not in model.layers
  ]
  # TODO(b/162887610): This may need to adjust the inbound node index if the
  # created layers had already been used to define other models.
  if ancillary_layers:
    new_nodes = nest.flatten([
      layer.inbound_nodes[1:]
      if functional._should_skip_first_node(layer)
      else layer.inbound_nodes for layer in created_layers.values()
    ])
    _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes)
  return model


def _insert_ancillary_layers(model, ancillary_layers, metrics_names, new_nodes):
  """Inserts ancillary layers into the model with the proper order."""
  # Sort `AddMetric` layers so they agree with metrics_names.
  metric_layers = [
    layer for layer in ancillary_layers if isinstance(layer, AddMetric)
  ]
  metric_layers.sort(key=lambda layer: metrics_names.index(layer.metric_name))
  ancillary_layers = [
                       layer for layer in ancillary_layers if not isinstance(layer, AddMetric)
                     ] + metric_layers
  model._insert_layers(ancillary_layers, relevant_nodes=list(new_nodes))


def _clone_layers_and_model_config(model, input_layers, layer_fn):
  """Clones all layers, and returns the model config without serializing layers.

  This function ensures that only the node graph is retrieved when getting the
  model config. The `layer_fn` used to clone layers might not rely on
  `layer.get_config()`, so some custom layers do not define `get_config`.
  Trying to retrieve the config results in errors.

  Args:
    model: A Functional model.
    input_layers: Dictionary mapping input layers in `model` to new input layers
    layer_fn: Function used to clone all non-input layers.

  Returns:
    Model config object, and a dictionary of newly created layers.
  """
  created_layers = {}
  def _copy_layer(layer):
    # Whenever the network config attempts to get the layer serialization,
    # return a dummy dictionary.
    if layer in input_layers:
      created_layers[layer.name] = input_layers[layer]
    elif layer in model._input_layers:
      created_layers[layer.name] = InputLayer(**layer.get_config())
    else:
      created_layers[layer.name] = layer_fn(layer)
    return {}

  config = functional.get_network_config(
    model, serialize_layer_fn=_copy_layer)
  return config, created_layers


def _make_new_nodes(nodes_by_depth, layer_fn, layer_map, tensor_map):
  """Uses the layers in `layer_map` to make new nodes based on `nodes_by_depth`.

  Args:
    nodes_by_depth: Provides structure information to create new nodes.
    layer_fn: Function to clone layers.
    layer_map: Map from layers in `model` to new layers.
    tensor_map: Map from tensors in `model` to newly compute tensors.

  Returns:
    A set of new nodes. `layer_map` and `tensor_map` are updated.
  """
  # Iterated over every node in the reference model, in depth order.
  new_nodes = set()
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)
  for depth in depth_keys:
    nodes = nodes_by_depth[depth]
    for node in nodes:
      # Recover the corresponding layer.
      layer = node.outbound_layer

      # Get or create layer.
      if layer not in layer_map:
        new_layer = layer_fn(layer)
        layer_map[layer] = new_layer
        layer = new_layer
      else:
        # Reuse previously cloned layer.
        layer = layer_map[layer]
        # Don't call InputLayer multiple times.
        if isinstance(layer, InputLayer):
          continue

      # If all previous input tensors are available in tensor_map,
      # then call node.inbound_layer on them.
      if all(
              tensor in tensor_map for tensor in nest.flatten(node.input_tensors)):
        # Call layer.
        args = nest.map_structure(lambda t: tensor_map.get(t, t),
                                  node.call_args)
        kwargs = nest.map_structure(lambda t: tensor_map.get(t, t),
                                    node.call_kwargs)
        output_tensors = layer(*args, **kwargs)

        # Thread-safe way to keep track of what node was created.
        first_output_tensor = nest.flatten(output_tensors)[0]
        new_nodes.add(
          layer._inbound_nodes[first_output_tensor._keras_history.node_index])

        for x, y in zip(
                nest.flatten(node.output_tensors), nest.flatten(output_tensors)):
          tensor_map[x] = y
  return new_nodes


def _remove_ancillary_layers(model, layer_map, layers):
  """Removes and returns any ancillary layers from `layers` based on `model`.

  Ancillary layers are part of the model topology but not used to compute the
  model outputs, e.g., layers from `add_loss` and `add_metric`.

  Args:
    model: A Keras Model.
    layer_map: A map to from layers in the `model` to those in `layers`.
    layers: A list of all layers.

  Returns:
    Two lists of layers: (1) `layers` with the ancillary layers removed, and (2)
    the ancillary layers.
  """
  ancillary_layers = []  # Additional layers for computing losses and metrics.
  if not model._is_graph_network:
    return layers, ancillary_layers

  # Ancillary layers are those with depth < 0.
  depths = [depth for depth in model._nodes_by_depth.keys() if depth < 0]
  depths.sort(reverse=True)  # Order topologically from inputs to outputs.
  for depth in depths:
    for node in model._nodes_by_depth[depth]:
      ancillary_layers.append(layer_map[node.outbound_layer])

  return [l for l in layers if l not in ancillary_layers], ancillary_layers
