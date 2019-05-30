import argparse
import numpy as np
import matplotlib.pyplot as plt
from CartoonGan import CartoonGAN
from tensorflow.python.keras.models import load_model
from utils import InstanceNormalization, ReflectionPadding2D

# function to postprocess the image: make it to range [0, 1] in order to be plotted
def postprocess(img):
    img = img[...,::-1]
    img = img * 0.5 + 0.5
    return img

# load models
custom = {'InstanceNormalization': InstanceNormalization, 'ReflectionPadding2D': ReflectionPadding2D}
generator = load_model('CartoonGan_generator.h5', custom_objects=custom)
test_img = np.load('dataset/photo_imgs_npy/0.npy')
test_out = generator(test_img)

# input photo
fig, ax = plt.subplots()
plt.subplot(1, 5, 1)
plt.axis('off')
plt.title("Input photo")
plt.imshow(postprocess(test_img))

# output cartoonized photo
plt.subplot(1, 5, 2)
plt.axis('off')
plt.title("Output photo")
plt.imshow(postprocess(test_out))

plt.tight_layout()
plt.show()