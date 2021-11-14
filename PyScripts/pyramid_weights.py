import tensorflow as tf
import numpy as np

class PyramidWeights:
  def __init__(self, weight, size_x, size_y): 
    prob_weights = 1
    if weight > 0:
      sx = np.expand_dims(np.linspace(1.0 / size_x, 1, size_x, dtype=np.float32), 1)
      sy = np.expand_dims(np.linspace(1.0 / size_y, 1, size_y, dtype=np.float32), 0)

      prob_weights = np.minimum(np.minimum(sx, np.flipud(sx)),
                                np.minimum(sy, np.fliplr(sy)))
      prob_weights /= np.max(prob_weights)
      prob_weights = np.minimum(prob_weights, weight)
    
    self.prob_weights = prob_weights
