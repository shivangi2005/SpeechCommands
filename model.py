from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.applications import Xception



def preprocess(x):
  x = (x + 0.8) / 7.0
  x = K.clip(x, -5, 5)
  return x


def preprocess_raw(x):
  return x


Preprocess = Lambda(preprocess)

PreprocessRaw = Lambda(preprocess_raw)


def relu6(x):
  return K.relu(x, max_value=6)


def conv_1d_time_stacked_model(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([800, 20])(x)
  x = PreprocessRaw(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
    x = Conv1D(
        num_filters,
        k,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.00001))(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
    x = Conv1D(
        num_filters,
        k,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=l2(0.00001),
        use_bias=False)(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

  x = _context_conv(x, 32, 1)
  x = _reduce_conv(x, 48, 3)
  x = _context_conv(x, 48, 3)
  x = _reduce_conv(x, 96, 3)
  x = _context_conv(x, 96, 3)
  x = _reduce_conv(x, 128, 3)
  x = _context_conv(x, 128, 3)
  x = _reduce_conv(x, 160, 3)
  x = _context_conv(x, 160, 3)
  x = _reduce_conv(x, 192, 3)
  x = _context_conv(x, 192, 3)
  x = _reduce_conv(x, 256, 3)
  x = _context_conv(x, 256, 3)

  x = Dropout(0.3)(x)
  x = Conv1D(num_classes, 5, activation='softmax')(x)
  x = Reshape([-1])(x)

  model = Model(input_layer, x, name='conv_1d_time_stacked')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model


def Conv1D_mfcc(input_size=5880, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([294, 20])(x)
  #x = Preprocess(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
    x = Conv1D(
        num_filters,
        k,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.00001))(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
    x = Conv1D(
        num_filters,
        k,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=l2(0.00001),
        use_bias=False)(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

  x = _context_conv(x, 32, 1)
  x = _reduce_conv(x, 48, 3)
  x = _context_conv(x, 48, 3)
  x = _reduce_conv(x, 96, 3)
  x = _context_conv(x, 96, 3)
  x = _reduce_conv(x, 128, 3)
  x = _context_conv(x, 128, 3)
  x = _reduce_conv(x, 160, 3)
  x = _context_conv(x, 160, 3)


  x = Dropout(0.3)(x)
  x = Conv1D(num_classes, 11, activation='softmax')(x)
  x = Reshape([-1])(x)

  model = Model(input_layer, x, name='Conv1D_mfcc')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model

def Conv2D_mfcc(input_size=5880, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([60, 98,1])(x)
  x = Preprocess(x)

  def _reduce_conv(x, num_filters, k, strides=(2,2), padding='valid'):
    x = Conv2D(
        num_filters,
        k,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.00001))(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    x = MaxPool2D(pool_size=(3,3), strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=(1,1), padding='valid'):
    x = Conv2D(
        num_filters,
        k,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=l2(0.00001),
        use_bias=False)(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

  x = _reduce_conv(x, 32, (3,3))
  x = _reduce_conv(x, 48, (3,3))
  x = _reduce_conv(x, 64, (3,3))
  #x = _reduce_conv(x, 96, (3,3))
  


  x = Dropout(0.3)(x)
  x = Flatten()(x)
  x = Dense(256)(x)
  x = BatchNormalization()(x)  
  x = Activation(relu6)(x)  
  x = Dropout(0.3)(x)
  x = Dense(num_classes, activation='softmax')(x)
  #x = Reshape([-1])(x)

  model = Model(input_layer, x, name='Conv2D_mfcc')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model

def Conv1D_melspec(input_size=5880, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([257, 98])(x)
  x = Preprocess(x)

  def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
    x = Conv1D(
        num_filters,
        k,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.00001))(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
    x = Conv1D(
        num_filters,
        k,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=l2(0.00001),
        use_bias=False)(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

  x = _context_conv(x, 32, 1)
  x = _reduce_conv(x, 48, 3)
  x = _context_conv(x, 48, 3)
  x = _reduce_conv(x, 96, 3)
  x = _context_conv(x, 96, 3)
  x = _reduce_conv(x, 128, 3)
  x = _context_conv(x, 128, 3)
  #x = _reduce_conv(x, 160, 3)
  #x = _context_conv(x, 160, 3)


  x = Dropout(0.3)(x)
  x = Conv1D(num_classes, 26, activation='softmax')(x)
  x = Reshape([-1])(x)

  model = Model(input_layer, x, name='Conv1D_mfcc')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model

def Conv2D_spec(input_size=5880, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([257, 98,1])(x)
  x = Preprocess(x)

  def _reduce_conv(x, num_filters, k, strides=(2,2), padding='valid'):
    x = Conv2D(
        num_filters,
        k,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.00001))(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    x = MaxPool2D(pool_size=(3,3), strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=(1,1), padding='valid'):
    x = Conv2D(
        num_filters,
        k,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=l2(0.00001),
        use_bias=False)(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

  #x = _context_conv(x, 32, (1,1))
  x = _reduce_conv(x, 32, (3,3))
  #x = _context_conv(x, 48, (3,3))
  x = _reduce_conv(x, 64, (3,3))
  #x = _context_conv(x, 96, (3,3))
  x = _reduce_conv(x, 128, (3,3))
  #x = _context_conv(x, 128, (3,3))
  #x = _reduce_conv(x, 128, (3,3))
  #x = _context_conv(x, 160, (3,3))


  x = Dropout(0.3)(x)
  x = Flatten()(x)
  x = Dense(256)(x)
  x = BatchNormalization()(x)  
  x = Activation(relu6)(x)  
  x = Dropout(0.3)(x)
  x = Dense(num_classes, activation='softmax')(x)
  #x = Reshape([-1])(x)

  model = Model(input_layer, x, name='Conv2D_mfcc')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model

def xception_spec(input_size=5880, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([257, 98,1])(x)
  x = Preprocess(x)  
  
  x = Conv2D(3,(3,3),padding='same')(x)  
  
  xc = Xception(include_top=False, weights='imagenet', input_shape = (257, 98, 3))
  x = (xc)(x)
  
  #x = Flatten()(x)
  #x = Dense(256)(x)
  #x = BatchNormalization()(x)  
  #x = Activation(relu6)(x)  
  x = Dropout(0.3)(x)
  x = Dense(num_classes, activation='softmax')(x)
  #x = Reshape([-1])(x)

  model = Model(input_layer, x, name='xception_spec')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model

def Conv2D_raw(input_size=16000, num_classes=11):
  """ Creates a 1D model for temporal data.

  Note: Use only
  with compute_mfcc = False (e.g. raw waveform data).
  Args:
    input_size: How big the input vector is.
    num_classes: How many classes are to be recognized.

  Returns:
    Compiled keras model
  """
  input_layer = Input(shape=[input_size])
  x = input_layer
  x = Reshape([800, 20,1])(x)
  x = Preprocess(x)

  def _reduce_conv(x, num_filters, k, strides=(2,2), padding='valid'):
    x = Conv2D(
        num_filters,
        k,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.00001))(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    x = MaxPool2D(pool_size=(3,3), strides=strides, padding=padding)(x)
    return x

  def _context_conv(x, num_filters, k, dilation_rate=(1,1), padding='valid'):
    x = Conv2D(
        num_filters,
        k,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_regularizer=l2(0.00001),
        use_bias=False)(
            x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    return x

  x = _reduce_conv(x, 32, (3,3))
  x = _reduce_conv(x, 64, (3,3))
  #x = _reduce_conv(x, 128, (3,3))
  


  x = Dropout(0.3)(x)
  x = Flatten()(x)
  x = Dense(256)(x)
  x = BatchNormalization()(x)  
  x = Activation(relu6)(x)  
  x = Dropout(0.3)(x)
  x = Dense(num_classes, activation='softmax')(x)
  #x = Reshape([-1])(x)

  model = Model(input_layer, x, name='Conv2D_raw')
  model.compile(
      optimizer=keras.optimizers.Adam(lr=3e-4),
      loss=keras.losses.categorical_crossentropy,
      metrics=[keras.metrics.categorical_accuracy])
  return model

def speech_model(model_type, input_size, num_classes=11, *args, **kwargs):
  if model_type == 'conv_1d_time_stacked':
    return conv_1d_time_stacked_model(input_size, num_classes)
  elif model_type == 'Conv1D_mfcc':
    return Conv1D_mfcc(input_size, num_classes)
  elif model_type == 'Conv2D_mfcc':
    return Conv2D_mfcc(input_size, num_classes)
  elif model_type == 'Conv1D_melspec':
    return Conv1D_melspec(input_size, num_classes)
  elif model_type == 'Conv2D_spec':
    return Conv2D_spec(input_size, num_classes)
  elif model_type == 'xception_spec':
    return xception_spec(input_size, num_classes)
  elif model_type == 'Conv2D_raw':
    return Conv2D_raw(input_size, num_classes)
  else:
    raise ValueError('Invalid model: %s' % model_type)


def prepare_model_settings(label_count,
                           sample_rate,
                           clip_duration_ms,
                           window_size_ms,
                           window_stride_ms,
                           dct_coefficient_count,
                           num_log_mel_features,
                           output_representation='raw'):
  """Calculates common settings needed for all models."""
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  spectrogram_frequencies = 257
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

  if output_representation == 'mfcc':
    fingerprint_size = num_log_mel_features * spectrogram_length
  elif output_representation == 'raw':
    fingerprint_size = desired_samples
  elif output_representation == 'spec':
    fingerprint_size = spectrogram_frequencies * spectrogram_length
  elif output_representation == 'mfcc_and_raw':
    fingerprint_size = num_log_mel_features * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'spectrogram_frequencies': spectrogram_frequencies,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'num_log_mel_features': num_log_mel_features
  }
