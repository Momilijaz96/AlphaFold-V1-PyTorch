# Lint as: python3.
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Write contact map predictions to a tf.io.gfile.

Either write a binary contact map as an RR format text file, or a
histogram prediction as a pickle of a dict containing a numpy array.
"""

import numpy as np
import six.moves.cPickle as pickle
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

def save_distance_histogram(
    filename, probs, domain, sequence, min_range, max_range, num_bins):
  """Save a distance histogram prediction matrix as a pickle file."""
  dh_dict = {
      'min_range': min_range,
      'max_range': max_range,
      'num_bins': num_bins,
      'domain': domain,
      'sequence': sequence,
      'probs': probs.astype(np.float32)}
  save_distance_histogram_from_dict(filename, dh_dict)


def save_distance_histogram_from_dict(filename, dh_dict):
  """Save a distance histogram prediction matrix as a pickle file."""
  fields = ['min_range', 'max_range', 'num_bins', 'domain', 'sequence', 'probs']
  missing_fields = [f for f in fields if f not in dh_dict]
  assert not missing_fields, 'Fields {} missing from dictionary'.format(
      missing_fields)
  assert len(dh_dict['sequence']) == dh_dict['probs'].shape[0]
  assert len(dh_dict['sequence']) == dh_dict['probs'].shape[1]
  assert dh_dict['num_bins'] == dh_dict['probs'].shape[2]
  assert dh_dict['min_range'] >= 0.0
  assert dh_dict['max_range'] > 0.0
  with tf.io.gfile.GFile(filename, 'wb') as fw:
    pickle.dump(dh_dict, fw, protocol=2)
