import argparse, os, shutil
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
    parser.add_argument('--epochs', type=int, default=500, help='epoch size for the training')
    parser.add_argument('--gpu_num', type=int, default=4, help='gpu numbers available for parallel training')
    parser.add_argument('--image_channels', type=int, default=3, help='image channels')
    parser.add_argument('--image_size', type=int, default=256, help='image size for the model input')
    parser.add_argument('--init_epoch', type=int, default=30, help='epoch size for the initial training of generator')
    parser.add_argument('--log_dir', type=str, default='logs', help='logs directory')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for the Adam optimizer')
    parser.add_argument('--model_dir', type=str, default='pretrained_model', help='pretrained model directory')
    parser.add_argument('--weight', type=int, default=10, help='the weight for the vgg loss in loss function')

    return parser.parse_args()

# main function
def main():
	args = parse_args()
    if(not os.path.isdir(args.model_dir)):
        os.mkdir(args.model_dir)

    # create cartoongan object
	cartoongan = CartoonGAN(args)

	# train model
	cartoongan.compile_model()
	cartoongan.train()

if __name__ == '__main__':
    main()