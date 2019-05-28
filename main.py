import argparse, os
import tensorflow as tf
from tensorflow.python.util import deprecation
from CartoonGan import CartoonGAN
from utils import *
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# argument parser
def parse_args():
    desc = "Keras implementation of CartoonGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for the training')
    parser.add_argument('--epochs', type=int, default=2000, help='epoch size for the training')
    parser.add_argument('--image_channels', type=int, default=3, help='image channels')
    parser.add_argument('--image_size', type=int, default=256, help='image size for the model input')
    parser.add_argument('--log_dir', type=str, default='log', help='train or test ?')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for the Adam optimizer')
    parser.add_argument('--weight', type=int, default=10, help='the weight for the vgg loss in loss function')

    return parser.parse_args()

# main function
def main():
	args = parse_args()

	batch_generator = DataGenerator(img_size=args.image_size, batch_size=args.batch_size)
	cartoongan = CartoonGAN(args)

	# train model
	cartoongan.compile_model()
	cartoongan.train(batch_generator)


if __name__ == '__main__':
    main()