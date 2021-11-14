import tensorflow as tf
import numpy as np
import os, sys
import json
import collections

ProteinExample = collections.namedtuple('ProteinExample', (
        'length', 'sequence', 'inputs_1d', 'inputs_2d', 'targets', 'residue_indices', 'domain'))

NUM_RES = 'sequence_length_placeholder' 
FEATURES = {
    'aatype': ('float32', [NUM_RES, 21]),
    'alpha_mask': ('int64', [NUM_RES, 1]),
    'alpha_positions': ('float32', [NUM_RES, 3]),
    'beta_mask': ('int64', [NUM_RES, 1]),
    'beta_positions': ('float32', [NUM_RES, 3]),
    'between_segment_residues': ('int64', [NUM_RES, 1]),
    'chain_name': ('string', [1]),
    'deletion_probability': ('float32', [NUM_RES, 1]),
    'domain_name': ('string', [1]),
    'gap_matrix': ('float32', [NUM_RES, NUM_RES, 1]),
    'hhblits_profile': ('float32', [NUM_RES, 22]),
    'hmm_profile': ('float32', [NUM_RES, 30]),
    #'key': ('string', [1]),
    'mutual_information': ('float32', [NUM_RES, NUM_RES, 1]),
    'non_gapped_profile': ('float32', [NUM_RES, 21]),
    'num_alignments': ('int64', [NUM_RES, 1]),
    'num_effective_alignments': ('float32', [1]),
    'phi_angles': ('float32', [NUM_RES, 1]),
    'phi_mask': ('int64', [NUM_RES, 1]),
    'profile': ('float32', [NUM_RES, 21]),
    'profile_with_prior': ('float32', [NUM_RES, 22]),
    'profile_with_prior_without_gaps': ('float32', [NUM_RES, 21]),
    'pseudo_bias': ('float32', [NUM_RES, 22]),
    'pseudo_frob': ('float32', [NUM_RES, NUM_RES, 1]),
    'pseudolikelihood': ('float32', [NUM_RES, NUM_RES, 484]),
    'psi_angles': ('float32', [NUM_RES, 1]),
    'psi_mask': ('int64', [NUM_RES, 1]),
    'residue_index': ('int64', [NUM_RES, 1]),
    'resolution': ('float32', [1]),
    'reweighted_profile': ('float32', [NUM_RES, 22]),
    'sec_structure': ('int64', [NUM_RES, 8]),
    'sec_structure_mask': ('int64', [NUM_RES, 1]),
    'seq_length': ('int64', [NUM_RES, 1]),
    'sequence': ('string', [1]),
    'solv_surf': ('float32', [NUM_RES, 1]),
    'solv_surf_mask': ('int64', [NUM_RES, 1]),
    'superfamily': ('string', [1]),
}

FEATURE_TYPES = {k: v[0] for k, v in FEATURES.items()}
FEATURE_SIZES = {k: v[1] for k, v in FEATURES.items()}

def shape(feature_name, num_residues, features=None):
  """Get the shape for the given feature name.
  Args:
    feature_name: String identifier for the feature. If the feature name ends
      with "_unnormalized", theis suffix is stripped off.
    num_residues: The number of residues in the current domain - some elements
      of the shape can be dynamic and will be replaced by this value.
    features: A feature_name to (tf_dtype, shape) lookup; defaults to FEATURES.
  Returns:
    List of ints representation the tensor size.
  """
  features = features or FEATURES
  if feature_name.endswith('_unnormalized'):
    feature_name = feature_name[:-13]

  unused_dtype, raw_sizes = features[feature_name]
  replacements = {NUM_RES: num_residues}

  sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
  return sizes


def parse_tfexample(raw_data, features):
  """Read a single TF Example proto and return a subset of its features.
  Args:
    raw_data: A serialized tf.Example proto.
    features: A dictionary of features, mapping string feature names to a tuple
      (dtype, shape). This dictionary should be a subset of
      protein_features.FEATURES (or the dictionary itself for all features).
  Returns:
    A dictionary of features mapping feature names to features. Only the given
    features are returned, all other ones are filtered out.
  """
  feature_map = {
      k: tf.io.FixedLenSequenceFeature(shape=(), dtype=v[0], allow_missing=True)
      for k, v in features.items()
  }
  parsed_features = tf.io.parse_single_example(raw_data, feature_map)

  # Find out what is the number of sequences and the number of alignments.
  num_residues = tf.cast(parsed_features['seq_length'][0], dtype=tf.int32)

  # Reshape the tensors according to the sequence length and num alignments.
  for k, v in parsed_features.items():
    new_shape = shape(feature_name=k, num_residues=num_residues)
    # Make sure the feature we are reshaping is not empty.
    assert_non_empty = tf.assert_greater(
        tf.size(v), 0, name='assert_%s_non_empty' % k,
        message='The feature %s is not set in the tf.Example. Either do not '
        'request the feature or use a tf.Example that has the feature set.' % k)
    with tf.control_dependencies([assert_non_empty]):
      parsed_features[k] = tf.reshape(v, new_shape, name='reshape_%s' % k)

  return parsed_features


def create_tf_dataset(tf_record_filename, features):
  """Creates an instance of tf.data.Dataset backed by a protein dataset SSTable.
  Args:
    tf_record_filename: A string with filename of the TFRecord file.
    features: A list of strings of feature names to be returned in the dataset.
  Returns:
    A tf.data.Dataset object. Its items are dictionaries from feature names to
    feature values.
  """
  # Make sure these features are always read.
  required_features = ['aatype', 'sequence', 'seq_length']
  features = list(set(features) | set(required_features))
  features = {name: FEATURES[name] for name in features}

  tf_dataset = tf.data.TFRecordDataset(filenames=[tf_record_filename])
  tf_dataset = tf_dataset.map(lambda raw: parse_tfexample(raw, features))

  return tf_dataset

class FeatureProcessor():
  def _load_dataset(self, tf_record_filename):
    features_to_load = self.features + self.targets
    return create_tf_dataset(tf_record_filename, features_to_load)

  def _normalize(self, data_row):
    for feature_name in self.copy_normalized_features:
      data_row[feature_name + '_unnormalized'] = data_row[feature_name]
      
    range_epsilon = 1e-12 # This is what alphafold uses
    for feature_name, feature_value in data_row.items():
      if feature_name in self.normalized_features:
        feature_as_float = tf.cast(feature_value, dtype=tf.float32)
        train_mean = tf.cast(self.normalization_stats['mean'][feature_name], dtype=tf.float32)
        train_range = tf.sqrt(tf.cast(self.normalization_stats['var'][feature_name], dtype=tf.float32))
        normalized_feature = feature_as_float - train_mean
        normalized_feature = tf.where(train_range > range_epsilon, normalized_feature / train_range, normalized_feature)
        data_row[feature_name] = normalized_feature
    return data_row

  def _group_by_dim(self, data_row):
    inputs_1d = []
    inputs_2d = []

    for feature_name in self.features:
      feature_as_float = tf.cast(data_row[feature_name], dtype=tf.float32)
      dim = len(feature_as_float.shape) - 1
      if dim == 1:
        inputs_1d.append(feature_as_float)
      elif dim == 2:
        inputs_2d.append(feature_as_float)
      else:
        raise f'Data type {feature_name} was not 1D or 2D '

    inputs_1d = tf.concat(inputs_1d, axis=1, name='inputs_1d_concat')
    inputs_2d = tf.concat(inputs_2d, axis=2, name='inputs_2d_concat')

    sequence = data_row['sequence'][0]
    sequence_length = tf.strings.length(data_row['sequence'])[0]
    
    targets = []
    for target_name in self.targets:
      targets.append(data_row.get(target_name + '_unnormalized', data_row[target_name]))

    if 'residue_index' in data_row:
      residue_indices = tf.squeeze(tf.cast(data_row['residue_index'], dtype=tf.int32), axis=1)
    else:
      residue_indices = tf.range(sequence_length)

    domain = data_row['domain_name'][0] if 'domain_name' in data_row else 'Unknown'

    target_class = collections.namedtuple('_TargetClass', self.targets)
    targets = target_class(*targets)

    return ProteinExample(length=sequence_length, sequence=sequence,
      inputs_1d=inputs_1d, inputs_2d=inputs_2d, targets=targets, residue_indices=residue_indices,
      domain=domain)
    
  def process_load(self):
    # Load the normalization statstics files (see text below this code cell)
    with tf.io.gfile.GFile(self.stats_file_path, 'r') as f:
      self.normalization_stats = json.load(f)
    
    return self._load_dataset(self.tf_record_filename)

  def process_normalize(self, dataset):
    return dataset.map(self._normalize)

  def process_all(self):
    dataset = self.process_load()
    dataset = self.process_normalize(dataset)
    dataset = dataset.map(self._group_by_dim)
    
    return dataset

  def __init__(self, config, tf_record_filename, stats_file_path):
    self.stats_file_path = stats_file_path
    self.tf_record_filename = tf_record_filename
    network_config = config['network_config']
    self.processed = False
    self.features = network_config['features']
    self.targets = network_config['targets']
    if network_config.get('is_ca_feature'):
      Raise('is_ca_feature needs to be implemented')
    self.normalized_features = list(set(self.features) - set(config['normalization_exclusion']))  # Calculate the features to normalize (all features but the specified excluded features)
    self.copy_normalized_features = list(set(self.features) & set(self.targets)) + ['num_alignments'] # The number of alignments is a required unnormalized feature


