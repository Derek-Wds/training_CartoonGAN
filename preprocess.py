import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# preprocess image
def preprocess(path, size=256, channels=3):
    # prepare
    sess = tf.Session()
    file_list = os.listdir(path)
    save_path = path + '_npy'

    with sess.as_default():
        for f in tqdm(file_list):
            file_name = os.path.basename(f)
            for i in ['.jpg', '.jpeg', '.png']:
                file_name  = file_name.rstrip(i)

            # start preprocess the image
            x = tf.read_file(path+'/'+f)
            x_decode = tf.image.decode_jpeg(x, channels=channels)
            img = tf.image.resize_images(x_decode, [size, size])
            img = tf.cast(img, tf.float32) / 127.5 - 1

            # save the preprocessed image
            np.save(os.path.join(save_path, file_name), img.eval())

if __name__ == "__main__":
    # make folders
    os.mkdir('dataset/photo_imgs_npy')
    os.mkdir('dataset/cartoon_imgs_npy')
    os.mkdir('dataset/smooth_cartoon_imgs_npy')

    # start preprocessing
    preprocess('dataset/photo_imgs')
    preprocess('dataset/cartoon_imgs')
    preprocess('dataset/smooth_cartoon_imgs')