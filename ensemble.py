import six.moves.cPickle as pickle

from distogram_io import save_distance_histogram_from_dict
files = [
    'predictions/full_0_e2.pickle',
    'predictions/full_1_e2.pickle',
    'predictions/full_2_e2.pickle',
    'predictions/full_3_e2.pickle',
    ]
out_file = 'predictions/ensemble_e2.pickle'

distos = []

for filename in files:
    with open(filename, 'rb') as f:
        distos.append(pickle.load(f, encoding="latin"))


for d in distos[1:]:
    distos[0]['probs'] += d['probs']

distos[0]['probs'] /= len(distos)


save_distance_histogram_from_dict(out_file, distos[0])