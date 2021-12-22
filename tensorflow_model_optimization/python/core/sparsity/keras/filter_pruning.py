from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import backend, Sequential
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.util import nest
from tensorflow.python.keras.engine import functional
from kerassurgeon import Surgeon

from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import dbg
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

layers = tf.keras.layers
keras = tf.keras


def _can_broadcast(shape1, shape2):
  """Whether two shapes can broadcast together using numpy rules"""
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
  """ Reduce the kernel (w, h, in_channels, out_channels) -> (out_channels,) boolean mask """
  # Reduce using max pooling
  kernel_reduced_to_filters = tf.reduce_max(kernel, axis=[0, 1, 2])

  # Boolean mask
  filter_mask_present = tf.greater(tf.abs(kernel_reduced_to_filters), 1e-20)

  # new_out_channels = tf.reduce_sum(tf.cast(filter_mask_present, tf.int32))

  return filter_mask_present


def _make_filter_indices(kernel):
  boolean_mask = tf.logical_not(_make_filter_mask(kernel))
  indices = tf.squeeze(tf.where(boolean_mask))
  indices_list = indices.numpy().tolist()
  return indices_list


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


def simplify_stripped_model(model):
  print('Simplifying model')
  # Assumes no pruning wrappers
  surgeon = Surgeon(model, copy=False)  # Assumes the stripping process has copied already

  dbg(model.layers)

  for layer in model.layers:
    if isinstance(layer, keras.layers.Conv2D):
      filters_to_remove = _make_filter_indices(layer.kernel)
      dbg(filters_to_remove)
      if filters_to_remove:
        surgeon.add_job('delete_channels', layer, channels=filters_to_remove)

  simplified_model = surgeon.operate()
  dbg(simplified_model)

  return simplified_model


# TODO(marcelroed): remove
# def strip_pruning_with_simplification(model):
#   """Strip pruning wrappers from the model, checking if output dimensions can be reduced.
#   Propagate changes to later layers.
#
#   Returns:
#       A keras model with pruning wrappers removed.
#   """
#
#   # TODO(marcelroed): Test if this works for an arbitrary DAG (functional model)
#
#   def _strip_pruning_wrapper(layer):
#     if isinstance(layer, tf.keras.Model):
#       # A keras model with prunable layers
#       raise ValueError('Cannot simplify nested models. Please flatten the model first (remove nested models).')
#       # return strip_pruning_with_simplification(model)
#
#     if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
#       # The _batch_input_shape attribute in the first layer makes a Sequential
#       # model to be built. This makes sure that when we remove the wrapper from
#       # the first layer the model's built state preserves.
#       if not hasattr(layer.layer, '_batch_input_shape') and hasattr(
#               layer, '_batch_input_shape'):
#         layer.layer._batch_input_shape = layer._batch_input_shape
#       return layer.layer
#     return layer
#
#   surgeon = Surgeon(model=model)
#
#   def _strip_pruning_remember_filter_pruned(model):
#     for layer in model.layers:
#       if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
#         if isinstance(layer.layer, keras.layers.Conv2D):
#           filters_to_remove = _make_filter_indices(layer.layer.kernel)
#           surgeon.add_job('remove_filters', layer.layer, channels=filters_to_remove)
#
#   new_model = surgeon.operate()
#
#   return keras.models.clone_model(
#     model, input_tensors=None, clone_function=_strip_pruning_wrapper)


