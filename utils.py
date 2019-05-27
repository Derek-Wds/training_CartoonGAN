import cv2, glob
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import Sequence

# data generator to get batches of data
class DataGenerator(Sequence):
    def __init__(self, img_size=256, batch_size=32, shuffle=True):
        self.photo_imgs = glob.glob("dataset/photo_imgs/*.*")
        self.cartoon_imgs = glob.glob("dataset/cartoon_imgs/*.*")
        self.smooth_cartoon_imgs = glob.glob("dataset/smooth_cartoon_imgs/*.*")
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    # return the length
    def __len__(self):
        length = min(len(self.photo_imgs),len(self.cartoon_imgs),len(self.smooth_cartoon_imgs))
        return int(length /self.batch_size)

    # the things to be returned at each batch
    def __getitem__(self, index):
        photo_batch = self.photo_imgs[index*self.batch_size: (index+1)*self.batch_size]
        cartoon_batch = self.cartoon_imgs[index*self.batch_size: (index+1)*self.batch_size]
        smooth_cartoon_batch = self.smooth_cartoon_imgs[index*self.batch_size: (index+1)*self.batch_size]
        
        return load(photo_batch), load(cartoon_batch), load(smooth_cartoon_batch)

    # shuffle at the epoch's end
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.photo_imgs)
            np.random.shuffle(self.cartoon_imgs)
            np.random.shuffle(self.smooth_cartoon_imgs)

# load image in the data generator
def load(imgs):
    ouput = np.array([preprocess(img) for img in imgs])
    return ouput

# preprocess image
def preprocess(filename, size=256, channels=3):
    x = tf.read_file(filename)
    x_decode = tf.image.decode_jpeg(x, channels=channels)
    img = tf.image.resize_images(x_decode, [size, size])
    img = tf.cast(img, tf.float32) / 127.5 - 1
    
    return img.eval()

# instance normalization implementation in keras
'''
This class is modified from the official github repo of Keras:
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py
'''
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# function to write the logs
def write_log(callback, name, value, batch):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch)
    callback.writer.flush()