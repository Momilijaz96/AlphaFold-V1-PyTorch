import tensorflow as tf
import numpy as np
from pyramid_weights import PyramidWeights
from data_augmenter import DataAugmenter
from cropper import Cropper
import matplotlib.pyplot as plt
from distogram_io import save_distance_histogram

class Predictor():
  def __init__(self, feature_processor, config, network):
    self.config = config
    self.feature_processor = feature_processor
    self.network = network

  def predict_all(self):
    config = self.config
    network_config = config['network_config']
    dataset = self.feature_processor.process_all()

    pyramid_weights = PyramidWeights(config['eval_config']['pyramid_weights'], config['crop_size_x'], config['crop_size_y'])
    augmenter = DataAugmenter(config['crop_size_x'], config['crop_size_y'], network_config['binary_code_bits'], name='Features2D')

    num_bins=network_config['num_bins']

    threshold = 8.0 # Find the bin that is 8A+
    # Note that this misuses the max_range as the range.
    bin_with_threshold = int(threshold - float(network_config['min_range']) * num_bins / float(network_config['max_range']))
    for protein_batch in dataset.batch(1): 
      length = protein_batch.length[0]
      crop_count = 0
      prob_accum = np.zeros([length, length])
      weight_accum = np.zeros([length, length])
      softmax_prob_accum = np.zeros([length, length, num_bins], dtype=np.float32)
      cropper = Cropper(config, protein_batch)
      crop_count_estimate = cropper.get_crop_count()
      print(f'Processing example with length: {length}, Number of crops: {crop_count_estimate}')
      for crop in cropper:
        print(f'Processing Crop {crop_count + 1} / {crop_count_estimate}')

        # Convert crop to "batch style"
        crop_x = tf.constant([crop.rectangle.x, crop.rectangle.end_x], dtype=tf.int32, shape=[1, 2])
        crop_y = tf.constant([crop.rectangle.y, crop.rectangle.end_y], dtype=tf.int32, shape=[1, 2])
            
        augmentation_features = augmenter(protein_batch.inputs_1d, crop.inputs_2d, protein_batch.residue_indices, crop_x, crop_y)

        # Todo: run network; for now simulate the nework output
        activations = self.network(augmentation_features, crop_x, crop_y)

        softmax_prob = tf.nn.softmax(activations[0, :, :, :num_bins])
        prob = tf.reduce_sum(softmax_prob[:, :, :bin_with_threshold], axis=2)

        weight = crop.get_live_data_hw(pyramid_weights.prob_weights)
        
        prob = crop.get_live_data_hw(prob)
        softmax_prob = crop.get_live_data_hwc(softmax_prob)

        prob *= weight
        softmax_prob *= tf.expand_dims(weight, axis=2)

        crop.stitch_data_onto_hw(prob, prob_accum)
        crop.stitch_data_onto_hw(weight, weight_accum)
        crop.stitch_data_onto_hwc(softmax_prob, softmax_prob_accum)

        crop_count += 1
      assert (weight_accum > 0.0).all()

      probs = prob_accum / weight_accum
      softmax_probs = softmax_prob_accum / tf.expand_dims(weight_accum, axis=2)

      # Add the transpose so prob(i, j) = prob (j, i)
      probs = (probs + probs.transpose()) / 2
      if num_bins > 1:
        softmax_probs = (softmax_probs + np.transpose(softmax_probs, axes=[1, 0, 2])) / 2

      filename = 'predictions/out.pickle'
      save_distance_histogram(filename,
        probs = softmax_probs.numpy(),
        sequence = protein_batch.sequence.numpy()[0],
        num_bins = num_bins,
        domain = protein_batch.domain.numpy()[0],
        min_range = float(network_config['min_range']),
        max_range = float(network_config['max_range']),
      )

      pred_contacts = np.sum(softmax_probs[:, :, :bin_with_threshold], axis=-1)
      plt.imshow(pred_contacts)
      plt.show()