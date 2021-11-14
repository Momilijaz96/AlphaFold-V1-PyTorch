import tensorflow as tf
import numpy as np

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

def get_feature(data):
    if isinstance(data, str):
        data = tf.constant(data, dtype=tf.string)
        
    if isinstance(data, np.ndarray):
        data = tf.constant(data)

    if data.dtype == tf.string:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.numpy()]))
    elif data.dtype == tf.int64:
        flattened = tf.reshape(data, [-1])
        return tf.train.Feature(int64_list=tf.train.Int64List(value=flattened.numpy()))
    elif data.dtype == tf.float32:
        flattened = tf.reshape(data, [-1])
        return tf.train.Feature(float_list=tf.train.FloatList(value=flattened.numpy()))
    else:
        raise 'Unknown data type'


class FeatureDataset:
    def __init__(self, domain):
        self.features = {}
        self.domain = domain

    def add_feature(self, name, value):
        self.features[name] = value

    def save_file(self, filename):
        # feature_map = {
        #     k: tf.io.FixedLenSequenceFeature(shape=(), dtype=v[0], allow_missing=True)
        #     for k, v in features.items()
        # }
        features = {name:get_feature(f) for (name, f) in self.features.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        record_bytes = example_proto.SerializeToString()

        with tf.io.TFRecordWriter(filename) as file_writer:
            file_writer.write(record_bytes)