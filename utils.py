import cv2, glob
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from vgg19 import VGG19

# data generator to get batches of data
class DataGenerator(keras.utils.Sequence):
    def __init__(self, root, img_size=256, batch_size=32, shuffle=True):
        self.root = root
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
        photo_batch = self.photo_imgs[idx*self.batch_size: (idx+1)*self.batch_size]
        cartoon_batch = self.cartoon_imgs[idx*self.batch_size: (idx+1)*self.batch_size]
        smooth_cartoon_batch = self.smooth_cartoon_imgs[idx*self.batch_size: (idx+1)*self.batch_size]

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

# function to write the logs
def write_log(callback, name, value, batch):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch)
    callback.writer.flush()