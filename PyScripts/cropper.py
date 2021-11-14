import tensorflow as tf
import numpy as np
import os, sys
import collections

Rectangle = collections.namedtuple('Rectangle', ('x', 'y', 'size_x', 'size_y', 'end_x', 'end_y'))
SingleCrop = collections.namedtuple('SingleCrop', ('rectangle', 'inputs_2d'))

class SingleCrop:
  def __init__(self, rectangle, batch): 
    self.rectangle = rectangle
    self.batch = batch
    self.length = self.batch.length[0]

    # Calculate the region of the input data
    self.data_x = max(0, rectangle.x)
    self.data_y = max(0, rectangle.y)
    self.data_end_x = min(self.length, rectangle.end_x)
    self.data_end_y = min(self.length, rectangle.end_y)

    # Calculate padding needed
    self.prepad_x = max(0, -rectangle.x)
    self.prepad_y = max(0, -rectangle.y)
    self.postpad_x = rectangle.end_x - self.data_end_x
    self.postpad_y = rectangle.end_y - self.data_end_y

    self._pad_and_generate_diagonals_2d()

  def _pad_and_generate_diagonals_2d(self): 
    inputs_2d = self.batch.inputs_2d
    rectangle = self.rectangle

    def pad_nhwc(input, pad_h, pad_w):
      return np.pad(input, [[0, 0], pad_h, pad_w, [0, 0]], mode='constant')
    
    # Pad the input 2D data; Input data is of form 'NHWC'
    inputs_2d = self.batch.inputs_2d[ :, self.data_y:self.data_end_y, self.data_x:self.data_end_x, :]
    inputs_2d = pad_nhwc(inputs_2d, [self.prepad_y, self.postpad_y], [self.prepad_x, self.postpad_x])

    # Generate diagonal data. That is data that encodes the local structure of both amino acids in the sequence
    # Generate two diagonals: one for the first residue (x) and one for the second residue (y)
    cxx = self.batch.inputs_2d[:, self.data_x:self.data_end_x, self.data_x:self.data_end_x, :]
    cyy = self.batch.inputs_2d[:, self.data_y:self.data_end_y, self.data_y:self.data_end_y, :]
    if cxx.shape[1] < inputs_2d.shape[1]:
      diagonal_offset_x = max(0, rectangle.x + rectangle.size_y - self.length)
      cxx = pad_nhwc(cxx, [self.prepad_x, diagonal_offset_x],[self.prepad_x, self.postpad_x])
    if cyy.shape[2] < inputs_2d.shape[2]:
      diagonal_offset_y = max(0, rectangle.y + rectangle.size_x - self.length)
      cyy = pad_nhwc(cyy, [self.prepad_y, self.postpad_y], [self.prepad_y, diagonal_offset_y])

    self.inputs_2d = np.concatenate([inputs_2d, cxx, cyy], 3)

  def stitch_data_onto_hwc(self, from_live_input, onto_quilt_output):
    onto_quilt_output[self.data_y:self.data_end_y, self.data_x:self.data_end_x, :] += from_live_input

  def get_live_data_hwc(self, data):
    return data[self.prepad_y:self.rectangle.size_y - self.postpad_y,
                self.prepad_x: self.rectangle.size_x - self.postpad_x, :]
                
  def stitch_data_onto_hw(self, from_live_input, onto_quilt_output):
    onto_quilt_output[self.data_y:self.data_end_y, self.data_x:self.data_end_x] += from_live_input

  def get_live_data_hw(self, data):
    return data[self.prepad_y:self.rectangle.size_y - self.postpad_y,
                self.prepad_x: self.rectangle.size_x - self.postpad_x]
class Cropper():
  def pos_generator(self):
    # Generate the "moving" crop positions
    for i in range(-self.crop_size_x // 2, self.length - self.crop_size_x // 2, self.crop_step_x):
      for j in range(-self.crop_size_y // 2, self.length - self.crop_size_y // 2, self.crop_step_y):
        yield (i, j)

  def __init__(self, config, protein_batch):
    self.batch = protein_batch
    self.length = self.batch.length[0]
    self.crop_size_x = config['crop_size_x']
    self.crop_size_y = config['crop_size_y']
    self.crop_step_x = self.crop_size_x // config['eval_config']['crop_shingle_x']
    self.crop_step_y = self.crop_size_y // config['eval_config']['crop_shingle_y']
    self.crop_positions = list(self.pos_generator())
    self.pos_gen = iter(self.crop_positions) 

  def get_crop_count(self):
    return len(self.crop_positions)

  def __iter__(self):
    return self

  def __next__(self):
    i, j = next(self.pos_gen)

    crop_rectangle = Rectangle(
        x=i, y=j, size_x=self.crop_size_x, size_y=self.crop_size_x,
        end_x=i+self.crop_size_x, end_y=j+self.crop_size_y)
    
    crop = SingleCrop(rectangle=crop_rectangle, batch=self.batch)
    return crop
