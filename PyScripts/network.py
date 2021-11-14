# Import libraries
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
# Activation and Regularization
from keras.regularizers import l2
from keras.activations import softmax

# Keras layers
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Activation, Add
#Imoprt Tensforflow and json for configuration
import json
import tensorflow as tf

# Avoid python call depth errors
import sys
sys.setrecursionlimit(5000)

def map_alphafold_weight_to_keras(alphafold_name):
  conv_weight_map = { 'b': 'conv2d/bias', 'w': 'conv2d/kernel'}
  conv_name_map = { '1x1': 'c_up', '1x1h': 'c_down', '3x3h': 'c_dia'}

  if alphafold_name == 'position_specific_bias':
    return 'position_bias/b'

  path = alphafold_name.split('/')
  if path[0] not in ['Deep2D', 'Deep2DExtra']: # todo: positions specific bias
    return None

  keras_name = path[0] + '/' # Keep the first part

  next = path[1]
  if next.startswith('res'):
    res_split = next.split('_')
    keras_name += res_split[0] + '/'
    if len(res_split) == 1:
      keras_name += 'batch_norm/' + path[2]
    else:
      conv_name = conv_name_map[res_split[1]]
      keras_name += conv_name + '/'
      if path[1] == path[2]:
        keras_name += 'batch_norm/' + path[3]
      else:
        conv_weight_name = conv_weight_map[path[2]]
        keras_name += conv_weight_name
  elif next.startswith('conv'):
    keras_name += next + '/'
    keras_name += conv_weight_map[path[2]]
  elif next.startswith('output_reshape'):
    keras_name += 'output_reshape/'
    if path[1] == path[2]:
      keras_name += 'batch_norm/' + path[3]
    else:
      keras_name += conv_weight_map[path[2]]
  return keras_name

class AlphaFoldConvLayer(tf.keras.layers.Layer):
  """Creates a convolution layer followed by a batchnorm
  and elu layer, which can be turned off by setting corresponding bool to false."""
  def __init__(self, num_filters,
                    kernel_size,
                    non_linearity=True,
                    batch_norm=False,
                    atrou_rate=1,
                    name=None):
    super(AlphaFoldConvLayer, self).__init__()
    if name is not None:
      self._name=name

    if batch_norm: #Check BN layer and decide bias addition
      use_bias=False
    else:
      use_bias=True

    self.batch_norm = BatchNormalization(scale=False, momentum=0.999, fused=True, name='batch_norm', trainable=True) if batch_norm else None
    self.elu = Activation('elu') if non_linearity else None
    self.conv = Conv2D(num_filters,kernel_size,strides=1,padding='same',
      data_format='channels_last',kernel_initializer='random_normal',
      kernel_regularizer=l2(1e-4),dilation_rate=atrou_rate,use_bias=use_bias, name="conv2d")

  def call(self, x):
    x = self.conv(x)

    if self.batch_norm:
      x = self.batch_norm(x, training=True)

    if self.elu:
      x = self.elu(x)

    return x

class AlphaFoldResBlock(tf.keras.layers.Layer):
  def __init__(self,
              num_filters,
              kernel_size,
              batch_norm=False,
              atrou_rate=1,
              dropout_keep_prob=1.0,
              name=None):
    """ Make a residual block
    Arguments:
        num_filters (int): Conv2D number of filters, same as channels of input/output of block
        kernel_size (int): Conv2D square kernel dimensions
        batch_norm (bool): whether to include batch normalization
        atrou_rate (int): dilation rate for the main(3x3 dilated) conv layer of block
    Return:
        A residual block output tensor
    """
    super(AlphaFoldResBlock, self).__init__()
    if name is not None:
      self._name = name

    self.batch_norm = BatchNormalization(scale=False, momentum=0.999, fused=True, name='batch_norm') if batch_norm else None
    self.elu = Activation('elu')

    #Downsize to half using a 1x1 conv
    self.conv_down = AlphaFoldConvLayer(num_filters//2,1,non_linearity=True,batch_norm=True,name='c_down')

    #3x3 dilated convolution layer
    self.conv_dilated = AlphaFoldConvLayer(num_filters//2,kernel_size,atrou_rate=atrou_rate,non_linearity=True,batch_norm=True,name='c_dia')

    #Upsize to half using a 1x1 conv
    #Note: We use TransposeConv2D for upsampling in Keras
    #x=Conv2DTranspose(num_filters,1,padding='same')(x)
    self.conv_up = AlphaFoldConvLayer(num_filters, 1, False,name='c_up')

    #Dropout
    self.dropout = Dropout(1-dropout_keep_prob) if dropout_keep_prob<1.0 else None

    #Skip connection
    self.skip_connect = Add()

  def call(self, input_node):
    x = input_node

    if self.batch_norm:
      x = self.batch_norm(x, training=True)

    x = self.elu(x)
    x = self.conv_down(x)
    x = self.conv_dilated(x)
    x = self.conv_up(x)

    if self.dropout:
      x = self.dropout(x)

    x = self.skip_connect([x,input_node])
    return x

class AlphaFoldResBlockStack(tf.keras.layers.Layer):
  def __init__(self,
    num_features=40,
    num_predictions=1,
    num_channels=32,
    num_blocks=2,
    filter_size=3,
    batch_norm=False,
    atrou_rates=None,
    #channel_multiplier=0,
    #divide_channels_by=2,
    dropout_keep_prob=1.0,
    name=None):
    """
      Make a stack of residual blocks with a conv layer at start and end.
    Arguments:
      input_node (tensor): from previous layer or input
      num_features (int): number of input channels
      num_predictions (int):number of channels of final output layer
      num_channels (int):Input and output number of channels of 
                          a single residual block
      num_blocks (int):number of residual blocks to stack + 2 conv layers
      filter_size (int):size of filter for main conv layer of each residual block
      batch_norm (bool): wether to use batch norm in a block or not
      atrou_rates (int): dilation rates for each subsequent residual block
      dropout_keep_prob (double)= 1 - drop_rate for an optional dropout layer at end of each block
      resize_features_with_1x1 (bool): Make start and end conv layer 1x1 or not
    Returns:
      Output of num_blocks stacked residual blocks
    """
    super(AlphaFoldResBlockStack, self).__init__()

    if name is not None:
      self._name=name

    if atrou_rates is None: atrou_rates = [1]
    non_linearity=True
    num_filters=num_channels

    #Loop over num blocks to stack
    self.blocks = []
    for i_block in range(num_blocks):
      #Get the current block's dilation rate
      curr_atrou_rate=atrou_rates[i_block % len(atrou_rates)]
      
      #Add a conv layer for first and last block
      is_first_block = (i_block==0)
      is_last_block = (i_block==num_blocks-1)
      if is_first_block or is_last_block:
        #For last block set the output channel size
        if is_last_block:
          num_filters=num_predictions
        self.blocks.append(AlphaFoldConvLayer(num_filters,filter_size,non_linearity=non_linearity,atrou_rate=curr_atrou_rate,name=f'conv{i_block+1}'))
      #Add middle residual blocks
      else:
        self.blocks.append(AlphaFoldResBlock(num_filters,filter_size,batch_norm=batch_norm,
                        atrou_rate=curr_atrou_rate,
                        dropout_keep_prob=dropout_keep_prob, name=f'res{i_block+1}'))
    
  def call(self, x):
    for block in self.blocks:
      x = block(x)

    return x

class PositionBias(tf.keras.Model):
  def __init__(self, bias_size):
    super(PositionBias, self).__init__()
    self.bias_size = bias_size

  def build(self, input_shape):
    main_input_shape = input_shape[0]
    self.crop_size_x = main_input_shape[1]
    self.crop_size_y = main_input_shape[2]
    self.num_bins = main_input_shape[3]

    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(self.bias_size, self.num_bins), dtype=tf.float32), trainable=True, name='b')


  def call(self, inputs):
    x, crop_x, crop_y = inputs

    # These are required because all inputs are feed in as floats (at least with build())
    crop_x = tf.cast(crop_x, tf.int32)
    crop_y = tf.cast(crop_y, tf.int32)
  
    # Find the coordinate of the final distogram that is most off the diagonal 
    max_off_diag = tf.reduce_max(tf.maximum(
      tf.abs(crop_x[:, 1] - crop_y[:, 0]), 
      tf.abs(crop_y[:, 1] - crop_x[:, 0])))

    # Add padding to the bias 
    padded_bias_size = tf.maximum(self.bias_size, max_off_diag)
    biases = tf.concat([self.b, tf.tile(self.b[-1:, :], [padded_bias_size - self.bias_size, 1])], axis=0)
    
    # Add mirror image of the bias for below-diagonal 
    biases = tf.concat([tf.reverse(biases[1:, :], axis=[0]), biases], axis=0)

    # For each crop in the batch, find the off-diagonal coordinate for the [0, 0] on the crop
    start_diag = crop_x[:, 0:1] - crop_y[:, 0:1] 

    # The off-diagonal decreases as y increases 
    diag_increment = tf.expand_dims(-tf.range(0, self.crop_size_y), 0)

    # Determine the off diagonal for each row start [0, row] on the crop
    row_start_diag = tf.reshape(start_diag + diag_increment, [-1])

    # Get the bias index of the rows(negative diagonal are between the start and middle of the bias array)
    row_start_diag_bias_index = row_start_diag + padded_bias_size - 1

    # Build crop biases row by row
    cropped_biases = tf.map_fn(lambda i: biases[i:i+self.crop_size_x, :], elems=row_start_diag_bias_index, fn_output_signature=tf.float32)
    cropped_biases = tf.reshape(cropped_biases, [-1, self.crop_size_y, self.crop_size_x, self.num_bins])

    return x + cropped_biases

class AlphaFoldNetwork(tf.keras.Model):
  def __init__(self, config):
    """
      Go from input features to the distance predictions.
      Arguments:
        input_shape(3d input shape tuple): Get input shape to initialize placeholders
        config(python nested distionary): Network architecture configuration file
      Output:
        Model
                              
    """
    super(AlphaFoldNetwork, self).__init__()

    #Get model's configuration file
    network_2d_deep = config['network_2d_deep']
    output_dimension = config['num_bins']
    num_features = 1878

    ##### LET'S START ASSEMBLING THE MODEL #####
    #220 Residual blocks with dilated convolution, with dilation rates of 4 subsequent 
    #blocks as [1,2,4,8], we are calling these four stacked blocks as 4-block-group.
    #Making 7 4-group-blocks at start of network with channels size of a residual blocks as 256
    self.extra_blocks = AlphaFoldResBlockStack(
                                      num_features=num_features,
                                      num_predictions=2*network_2d_deep['num_filters'],
                                      num_channels= 2*network_2d_deep['num_filters'],
                                      num_blocks=network_2d_deep['extra_blocks'] * network_2d_deep['num_layers_per_block'],
                                      filter_size=3,
                                      batch_norm=network_2d_deep['use_batch_norm'],
                                      atrou_rates = [1,2,4,8],
                                      dropout_keep_prob=1.0,
                                      name='Deep2DExtra'
    ) if network_2d_deep['extra_blocks'] else None
        
    #doble input feature size for next half of the network
    num_features = 2 * network_2d_deep['num_filters']

    #Making 48 4-group-blocks at start of network with channels size of a residual blocks as 128
    self.main_blocks = AlphaFoldResBlockStack(
                                      num_features=num_features,
                                      num_predictions=network_2d_deep['num_filters'] if config['reshape_layer'] else output_dimension,
                                      num_channels= network_2d_deep['num_filters'],
                                      num_blocks=network_2d_deep['num_blocks'] * network_2d_deep['num_layers_per_block'],
                                      filter_size=3,
                                      batch_norm=network_2d_deep['use_batch_norm'],
                                      atrou_rates = [1,2,4,8],
                                      dropout_keep_prob=1.0,
                                      name='Deep2D'
                                      )
    #Add a 1x1 conv layer to resize the output contact_pre_logits
    #if config.reshape_layer was true then the contact_pre_logits output
    #is network_2d_deep.num_filters size sized so change it to num_bins
    self.reshape = AlphaFoldConvLayer(
          num_filters=output_dimension,
          kernel_size=1,
          non_linearity=False,
          batch_norm=network_2d_deep['use_batch_norm'],
          name='Deep2D/output_reshape'
    ) if config['reshape_layer'] else None

    num_biases = config['position_specific_bias_size']
    self.position_bias = PositionBias(num_biases) if num_biases else None

  @tf.function
  def call(self, inputs):
    x, crop_x, crop_y = inputs

    if self.extra_blocks:
      x = self.extra_blocks(x)

    x = self.main_blocks(x)

    if self.reshape:
      x = self.reshape(x)

    if self.position_bias:
      x = self.position_bias([x, crop_x, crop_y])
    
    return x

  def set_weights_from_old_alphafold(self, saved_model_path):
    weights = {k.name.split(':')[0]:None for k in self.weights}

    reader = tf.train.load_checkpoint(saved_model_path)
    shape_from_key = reader.get_variable_to_shape_map()
    for alphafold_weight_name in shape_from_key:
      name = map_alphafold_weight_to_keras(alphafold_weight_name)
      if name is None:
        continue

      weights[name] = reader.get_tensor(alphafold_weight_name)

    weight_values = list(weights.values())
    self.set_weights(weight_values)

## Network configurations

#Get the network configuration dictionary
config_file_path = 'alphafold-casp13-weights/873731/0/config.json'

# Load the first distogram replica config file
with tf.io.gfile.GFile(config_file_path, 'r') as f:
  config = json.load(f)
config

## Test network architecture

# Check it's working
if __name__ == "__main__":
  # Using AMSGrad optimizer for speed 
  kernel_size, filters = 3, 16
  adam = keras.optimizers.Adam(amsgrad=True)
  # Create model
  model = AlphaFoldNetwork(config=config['network_config'])
  model.build(input_shape=(1,64,64,1878))
  model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

  # # Keras model weights:
  # for w in model.weights:
  #   print(f'{w.name}: {w.shape}')

  # Sonnet model weights:
  saved_model_path = 'alphafold-casp13-weights/873731/0/tf_graph_data/tf_graph_data.ckpt'
  model.set_weights_from_old_alphafold(saved_model_path)

