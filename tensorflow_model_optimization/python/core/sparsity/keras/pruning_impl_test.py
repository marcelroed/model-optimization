# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the key functions in pruning library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.internal.tensor_encoding.utils import dbg

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_impl
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils

K = tf.keras.backend
dtypes = tf.dtypes
test = tf.test


def assign_add(ref, value):
  if hasattr(tf, "assign_add"):
    return tf.assign_add(ref, value)
  else:
    return ref.assign_add(value)


@keras_parameterized.run_all_keras_modes
class PruningTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PruningTest, self).setUp()
    self.block_size = (1, 1)
    self.block_pooling_type = "AVG"

    self.constant_sparsity = pruning_schedule.ConstantSparsity(0.5, 0, 100, 1)

  # Variable initialization outside of setUp() is needed for compatibility with
  # run_all_keras_modes.
  #
  # setUp() lies outside of the "eager scope" that wraps the test cases
  # themselves, resulting in initializing graph tensors instead of eager
  # tensors when testing eager execution.
  def initialize(self):
    """Set the global step to a tf.Variable containing 0 and the global step function to return the global step"""
    self.global_step = tf.Variable(
        tf.zeros([], dtype=dtypes.int32),
        dtype=dtypes.int32,
        name="global_step")

    def training_step_fn():
      return self.global_step
    self.training_step_fn = training_step_fn

    compat.initialize_variables(self)

  def testExtremelySparseMask(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        name="mask",
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
    self.initialize()

    extreme_sparsity = pruning_schedule.ConstantSparsity(0.9999, 0, 100, 1)
    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=extreme_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    mask_before_pruning = K.get_value(mask)
    self.assertAllEqual(np.count_nonzero(mask_before_pruning), 100)

    if tf.executing_eagerly():
      p.conditional_mask_update()
    else:
      K.get_session().run(p.conditional_mask_update())

    # We should always have a single connection remaining.
    mask_after_pruning = K.get_value(mask)
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 1)

  def testUpdateSingleMask(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        name="mask",
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
    self.initialize()

    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=self.constant_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    mask_before_pruning = K.get_value(mask)
    self.assertAllEqual(np.count_nonzero(mask_before_pruning), 100)

    if tf.executing_eagerly():
      p.conditional_mask_update()
    else:
      K.get_session().run(p.conditional_mask_update())

    mask_after_pruning = K.get_value(mask)
    self.assertAllEqual(np.count_nonzero(mask_after_pruning), 50)

  def testConstructsMaskAndThresholdCorrectly(self):
    self.initialize()
    p = pruning_impl.Pruning(
        lambda: 0, None,
        # Sparsity math often returns values with small tolerances.
        lambda x: (True, 0.200000018),
        (1, 1), None)

    # input matrix is [ 1.0, 2.0, ..., 8.0, 9.0, 10.0 ]
    threshold, mask = p._update_mask(np.arange(1, 11))

    self.assertEqual(3, K.get_value(threshold))
    self.assertAllEqual(
        # expected matrix is [ 0.0, 0.0, 1.0, 1.0 ... 1.0 ]
        np.concatenate((np.zeros(2), np.ones(8))), K.get_value(mask))

  def _blockMasking(self, block_size, block_pooling_type, weight,
                    expected_mask):
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight.dtype),
        name="mask",
        dtype=weight.dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight.dtype), name="threshold", dtype=weight.dtype)
    self.initialize()

    # Set up pruning
    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=self.constant_sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type)

    _, new_mask = p._maybe_update_block_mask(weight)
    # Check if the mask is the same size as the weights
    self.assertAllEqual(new_mask.get_shape(), weight.get_shape())
    mask_after_pruning = K.get_value(new_mask)
    self.assertAllEqual(mask_after_pruning, expected_mask)

  def testBlockMaskingAvg(self):
    block_size = (2, 2)
    block_pooling_type = "AVG"
    weight = tf.constant([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                          [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testBlockMaskingMax(self):
    block_size = (2, 2)
    block_pooling_type = "MAX"
    weight = tf.constant([[0.1, 0.0, 0.2, 0.0], [0.0, -0.1, 0.0, -0.2],
                          [0.3, 0.0, 0.4, 0.0], [0.0, -0.3, 0.0, -0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testBlockMaskingWithHigherDimensionsRaisesError(self):
    self.initialize()
    block_size = (2, 2)
    block_pooling_type = "AVG"
    # Weights as in testBlockMasking, but with one extra dimension.
    weight = tf.constant([[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                           [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]]])
    expected_mask = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                      [1., 1., 1., 1.], [1., 1., 1., 1.]]]

    # Block masking should only be used with 2 Dimensional weights.
    with self.assertRaises(ValueError):
      self._blockMasking(block_size, block_pooling_type, weight, expected_mask)

  def testConditionalMaskUpdate(self):
    weight = tf.Variable(np.linspace(1.0, 100.0, 100), name="weights")
    weight_dtype = weight.dtype.base_dtype
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        name="mask",
        dtype=weight_dtype)
    threshold = tf.Variable(
        tf.zeros([], dtype=weight_dtype), name="threshold", dtype=weight_dtype)
    self.initialize()

    def linear_sparsity(step):
      sparsity_val = tf.convert_to_tensor(
          [0.0, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
      return tf.convert_to_tensor(True), sparsity_val[step]

    # Set up pruning
    p = pruning_impl.Pruning(
        pruning_vars=[(weight, mask, threshold)],
        training_step_fn=self.training_step_fn,
        pruning_schedule=linear_sparsity,
        block_size=self.block_size,
        block_pooling_type=self.block_pooling_type)

    non_zero_count = []
    for _ in range(10):
      if tf.executing_eagerly():
        p.conditional_mask_update()
        p.weight_mask_op()
        assign_add(self.global_step, 1)
      else:
        K.get_session().run(p.conditional_mask_update())
        K.get_session().run(p.weight_mask_op())
        K.get_session().run(assign_add(self.global_step, 1))

      non_zero_count.append(np.count_nonzero(K.get_value(weight)))

    # Weights pruned at steps 1,3,5
    expected_non_zero_count = [100, 90, 90, 70, 70, 50, 50, 50, 50, 50]
    self.assertAllEqual(expected_non_zero_count, non_zero_count)

  def testFilterPruning(self):
    # (width, height, in_channels, out_channels)
    weight_shape = (1, 1, 2, 3)
    weight_size = np.prod(weight_shape)  # number of elements in the weight tensor
    # Construct weight matrix using numbers 1.0..weight_size
    weight = tf.Variable(np.linspace(1.0, float(weight_size), int(weight_size)).reshape(weight_shape), name="weights")
    weight_dtype = weight.dtype.base_dtype

    # Mask starts of as matrix of ones of the same shape as the weight matrix
    mask = tf.Variable(
        tf.ones(weight.get_shape(), dtype=weight_dtype),
        name="mask",
        dtype=weight_dtype)

    # Threshold starts off scalar tensor 0
    threshold = tf.Variable(
        tf.convert_to_tensor(0, dtype=weight_dtype), name="threshold", dtype=weight_dtype)

    self.initialize()  # Set self.global_step and self.global_step_fn

    # Pruning schedule
    def linear_sparsity(step):
      sparsity_val = tf.convert_to_tensor(
        [0.0, 0.1, 0.1, 0.3, 0.3, 0.51, 0.51, 0.51, 0.51, 0.51])
      # TODO(marcelroed) what does the bool mean?
      #      ?whether or not to prune?   sparsity proportion
      return tf.convert_to_tensor(True), sparsity_val[step]

    p = pruning_impl.Pruning(
      pruning_vars=[(weight, mask, threshold)],
      training_step_fn=self.training_step_fn,
      pruning_schedule=linear_sparsity,
      block_size=(1, 1),
      block_pooling_type='AVG',
      filter_pruning_description={'filter_block_pooling_type': 'MAX'}
    )
    non_zero_count = []
    tf.print('Before pruning:', self.global_step, K.get_value(weight), mask)
    for _ in range(10):
      p.conditional_mask_update()
      p.weight_mask_op()
      assign_add(self.global_step, 1)
      non_zero_count.append(np.count_nonzero(K.get_value(weight)))
      tf.print(self.global_step, K.get_value(weight))
      if self.global_step.read_value() < 10:
        tf.print(linear_sparsity(self.global_step.read_value()))

    # Why does it not prune 50% when the sparsity is at 0.5?
    # Because the output channels are being pruned, meaning we have to prune either 33.33%, 66.66% or 100%
    # If we change the sparsity to 0.51 we find that 66.66% is closer to the desired sparsity, so it will prune 2/3 output channels.
    dbg(non_zero_count)




if __name__ == "__main__":
  test.main()
