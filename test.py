import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from main import parse_args
from CartoonGan import CartoonGAN
from tensorflow.python.keras.models import load_model
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# function to postprocess the image: make it to range [0, 1] in order to be plotted
def postprocess(img):
    img = img / 2 + 0.5
    return img

# load models
cartoongan = CartoonGAN(parse_args())
cartoongan.compile_model()
cartoongan.generator.load_weights('pretrained_model/CartoonGan_generator_epoch_300.h5')

test_img = np.load('test.npy')
test_in = test_img.reshape(1, test_img.shape[0], test_img.shape[1], test_img.shape[2])
test_out = cartoongan.generator.predict(test_in)

test_out = test_out.reshape(test_out.shape[1], test_out.shape[2], test_out.shape[3])

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
