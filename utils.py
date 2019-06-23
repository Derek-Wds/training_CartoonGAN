import glob, random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import Sequence
from io import BytesIO

# data generator to get batches of data
class DataGenerator(Sequence):
    def __init__(self, image_size=256, batch_size=32, shuffle=True):
        self.photo_imgs = glob.glob("dataset/photo_imgs_npy/*.*")
        self.cartoon_imgs = glob.glob("dataset/cartoon_imgs_npy/*.*")
        self.smooth_cartoon_imgs = glob.glob("dataset/smooth_cartoon_imgs_npy/*.*")
        self.image_size = image_size
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
        
        return load(photo_batch), load(cartoon_batch), load(smooth_cartoon_batch), index

    # shuffle at the epoch's end
    def on_epoch_end(self):
        if self.shuffle == True:
            seed = random.randint(1, 6666)
            np.random.seed(seed) # set seed for cartoon images
            np.random.shuffle(self.photo_imgs)
            np.random.shuffle(self.cartoon_imgs)
            np.random.seed(seed) # make sure smoothed cartoon images are the same
            np.random.shuffle(self.smooth_cartoon_imgs)

# load numpy file
def load(file_list):
    output = np.array([np.load(img) for img in file_list])
    return output

# ReflectionPadding class
class ReflectionPadding2D(Layer):
    def __init__(self,
                 padding=(1, 1),
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        else:
            self.padding = ((1, 1), (1, 1))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
        else:
            rows = None
        if input_shape[2] is not None:
            cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
        else:
            cols = None
        return (input_shape[0], rows, cols, input_shape[3])

    def call(self, inputs):
        pattern = [[0, 0], list(self.padding[0]), list(self.padding[1]), [0, 0]]
        return tf.pad(inputs, pattern, "REFLECT")

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

# function to write images to logs
def write_images(callback, images, name, batch):
    number = 0
    img_summaries = []
    for i in images:
        s = BytesIO()
        plt.imsave(s, i, format='png')
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=i.shape[0],
                                       width=i.shape[1])
        img_summaries.append(tf.Summary.Value(tag='%s/%d' % (name, number),
                                                 image=img_sum))
        number += 1
    summary = tf.Summary(value=img_summaries)
    callback.writer.add_summary(summary, batch)
    callback.writer.flush()