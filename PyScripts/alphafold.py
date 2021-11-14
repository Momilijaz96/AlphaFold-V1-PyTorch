import tensorflow as tf
import numpy as np
import json
from feature_loader import FeatureProcessor
from predictor import Predictor
from network import AlphaFoldNetwork

stats_file_path = 'alphafold-casp13-weights/873731/stats_train_s35.json'
config_file_path = 'alphafold-casp13-weights/873731/0/config.json'
tf_record_filename = 'examples/T0958/T0958.tfrec'
saved_model_path = 'alphafold-casp13-weights/873731/0/tf_graph_data/tf_graph_data.ckpt'

def main():
  with tf.io.gfile.GFile(config_file_path, 'r') as f:
    config = json.load(f)
    network_config = config['network_config']

  feature_processor = FeatureProcessor(config, tf_record_filename, stats_file_path)

  def avg_network(inputs, crop_x, crop_y):
    prob = tf.constant(1, shape=[1, inputs.shape[1], inputs.shape[2], network_config['num_bins']], dtype=tf.float32)
    prob = prob + tf.pad(tf.math.reduce_mean(inputs, axis=3, keepdims=True), [[0,0], [0,0],[0,0], [0,network_config['num_bins'] - 1]])
    return prob

  network = AlphaFoldNetwork(network_config)

  def alphafold_network(inputs, crop_x, crop_y):
    if not network.built:
      network.build([inputs.shape, crop_x.shape, crop_y.shape])
      network.set_weights_from_old_alphafold(saved_model_path)
    activations = network([inputs, crop_x, crop_y])
    return activations

  predictor = Predictor(feature_processor, config, alphafold_network)
  predictor.predict_all()

if __name__ == '__main__':
  main()
