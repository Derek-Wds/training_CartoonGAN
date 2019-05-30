import cv2, glob
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

# data generator to get batches of data
class DataGenerator(Sequence):
    def __init__(self, img_size=256, batch_size=32, shuffle=True):
        self.photo_imgs = glob.glob("dataset/photo_imgs_npy/*.*")
        self.cartoon_imgs = glob.glob("dataset/cartoon_imgs_npy/*.*")
        self.smooth_cartoon_imgs = glob.glob("dataset/smooth_cartoon_imgs_npy/*.*")
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

# load numpy file
def load(file_list):
    output = np.array([np.load(img) for img in file_list])
    return output

# function to write the logs
def write_log(callback, name, value, batch):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch)
    callback.writer.flush()