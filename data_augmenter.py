import tensorflow as tf
import numpy as np

class DataAugmenter():
  def __init__(self, crop_size_x, crop_size_y, binary_code_bits, name=None):
    self.binary_code_bits = binary_code_bits
    self.crop_size_x = crop_size_x
    self.crop_size_y = crop_size_y

  def __call__(self, inputs_1d, inputs_2d_cropped, residue_index, crop_x, crop_y):
    l = tf.shape(inputs_1d)[1]
    n_x = self.crop_size_x
    n_y = self.crop_size_y

    def feature_crop(features, crop, size, dtype):
      def per_sample(s):
        crop, features = s
        cropped_1d = features[max(0, crop[0]):crop[1]]
        paddings = tf.concat([max(0, -crop[0]), max(0, size -(l - crop[0]))], axis=0)
        paddings = tf.expand_dims(paddings, axis=0)
        paddings = tf.pad(paddings, [[0, tf.rank(cropped_1d)-1], [0, 0]])
        return tf.pad(cropped_1d, paddings)
      return tf.map_fn(per_sample, elems=(crop, features), dtype=dtype)

    inputs_1d_cropped_y = feature_crop(inputs_1d, crop_y, n_y, dtype=tf.float32)
    range_n_y = feature_crop(residue_index, crop_y, n_y, dtype=tf.int32)

    inputs_1d_cropped_x = feature_crop(inputs_1d, crop_x, n_x, dtype=tf.float32)
    range_n_x = feature_crop(residue_index, crop_x, n_x, dtype=tf.int32)
    
    range_scale = 100.0  # "Crude normalization factor" Per deepmind

    offset = (tf.expand_dims(tf.cast(range_n_x, tf.float32), 1) -
              tf.expand_dims(tf.cast(range_n_y, tf.float32), 2)) / range_scale

    position_features = [
        tf.tile(tf.reshape((tf.cast(range_n_y, tf.float32) - range_scale) / range_scale,
                [-1, n_y, 1, 1]), [1, 1, n_x, 1]),
        tf.tile(tf.reshape(offset, [-1, n_y, n_x, 1]), [1, 1, 1, 1])
    ]

    if self.binary_code_bits:
      exp_range_n_y = tf.expand_dims(range_n_y, 2)
      bin_y = tf.stop_gradient(
          tf.concat([tf.math.floormod(exp_range_n_y // (1 << i), 2)
                      for i in range(self.binary_code_bits)], 2))
      exp_range_n_x = tf.expand_dims(range_n_x, 2)
      bin_x = tf.stop_gradient(
          tf.concat([tf.math.floormod(exp_range_n_x // (1 << i), 2)
                      for i in range(self.binary_code_bits)], 2))
      position_features += [
          tf.tile(tf.expand_dims(tf.cast(bin_y, tf.float32), 2), [1, 1, n_x, 1]),
          tf.tile(tf.expand_dims(tf.cast(bin_x, tf.float32), 1), [1, n_y, 1, 1])
      ]

      augmentation_features = position_features + [
          tf.tile(tf.expand_dims(inputs_1d_cropped_x, 1), [1, n_y, 1, 1]),
          tf.tile(tf.expand_dims(inputs_1d_cropped_y, 2), [1, 1, n_x, 1])
      ]
      hidden_2d = tf.concat([inputs_2d_cropped] + augmentation_features, 3)
      return hidden_2d