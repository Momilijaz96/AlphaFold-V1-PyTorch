import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import numpy as np
from sklearn.metrics import mean_squared_error

with open('/home/d/Desktop/T0958_Predictions/deepmind.pickle', 'rb') as f:
    distogram_dict = pickle.load(f, encoding="latin")

with open('/home/d/Desktop/T0958_Predictions/true_dis.pickle', 'rb') as f:
    true_dis = pickle.load(f, encoding="latin")['true_dis']

num_bins = distogram_dict['probs'].shape[-1]
bin_size_angstrom = distogram_dict['max_range'] / num_bins

def get_prob_less_distance(thres, probs):
    thres = max(distogram_dict['min_range'], thres)
    threshold_cts = (thres - distogram_dict['min_range']) / bin_size_angstrom
    threshold_bin = int(threshold_cts)  # Round down
    if threshold_bin >= len(probs):
        return 1.0
    pred_contacts = np.sum(probs[:threshold_bin], axis=-1)
    if threshold_bin < threshold_cts:  # Add on the fraction of the boundary bin.
        pred_contacts += probs[threshold_bin] * (
            threshold_cts - threshold_bin)
    return pred_contacts


"""Split the boundary bin."""
pred_contacts = np.vectorize(lambda p: get_prob_less_distance(8.0, p),signature='(n)->()')(distogram_dict['probs'])
pred_contacts = pred_contacts[:85, :85]
true_contacts = np.where(true_dis < 8.0, 1, 0)

# Predicted Distance
rel_probs = distogram_dict['probs'][:85, :85, :]
max_prob_bins = np.argmax(rel_probs, axis=2)
pred_dis = distogram_dict['min_range'] + bin_size_angstrom * max_prob_bins

def iddt(actual, pred, r):
    l = len(actual)
    relevant_err_mask = np.array([[1 if abs(i - j) >= r else 0 for j in range(l)] for i in range(l)])
    relevant_err_mask = np.multiply(np.where(actual < 15, 1, 0), relevant_err_mask)

    def get_prob_within_distance(t1, t2, probs):
        x1 = get_prob_less_distance(t1, probs)
        x2 = get_prob_less_distance(t2, probs)
        prob = x2 - x1
        return prob

    iddt_score_mat = np.zeros([l, l])
    for t in [0.5, 1, 2, 4]:
        err = np.vectorize(lambda a, p: get_prob_within_distance(a - t, a + t, p), signature='(),(m)->()')(actual, pred)
        iddt_score_mat += err

    iddt_score_mat = np.multiply(iddt_score_mat, relevant_err_mask)

    iddt_score = 100 * np.sum(iddt_score_mat) / np.sum(relevant_err_mask) / 4
    return iddt_score, relevant_err_mask

rmse = mean_squared_error(pred_contacts[:85, :85], true_contacts, squared=False)
iddt_12, relevant_err_mask = iddt(true_dis, rel_probs, 12)
#536.9444228777312

plt.subplot(1,3,1)
plt.imshow(true_contacts)
plt.title('T0958 True')
plt.subplot(1,3,2)
plt.imshow(pred_contacts)
plt.title('T0958 Our Prediction')
plt.subplot(1,3,3)
plt.imshow(1-np.abs(true_contacts - pred_contacts), cmap='gray', vmin=0, vmax=1)
plt.title('T0958 Diff')
plt.show()