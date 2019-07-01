import time, random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import multi_gpu_model
from utils import *

# class for CartoonGAN
class CartoonGAN():
    def __init__(self, args):
        self.model_name = 'CartoonGAN'
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.gpu = args.gpu_num
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.init_epoch = args.init_epoch
        self.log_dir = args.log_dir
        self.lr = args.lr
        self.model_dir = args.model_dir
        self.weight = args.weight
    
    # method for generator
    def generator(self):
        input_shape=[self.image_size, self.image_size, self.image_channels]
        input_img = Input(shape=input_shape, name="input")

        # first block
        x = ReflectionPadding2D(3)(input_img)
        x = Conv2D(64, (7, 7), strides=1, use_bias=True, padding='valid', name="conv1")(x)
        x = InstanceNormalization(name="norm1")(x)
        x = Activation("relu")(x)

        # down-convolution
        channel = 128
        for i in range(2):
            x = Conv2D(channel, (3, 3), strides=2, use_bias=True, padding='same', name="conv{}_1".format(i+2))(x)
            x = Conv2D(channel, (3, 3), strides=1, use_bias=True, padding='same', name="conv{}_2".format(i+2))(x)
            x = InstanceNormalization(name="norm{}".format(i+2))(x)
            x = Activation("relu")(x)
            channel = channel * 2
        
        # residual blocks
        x_res = x
        for i in range(8):
            x = ReflectionPadding2D(1)(x)
            x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv{}_1".format(i+4))(x)
            x = InstanceNormalization(name="norm{}_1".format(i+4))(x)
            x = Activation("relu")(x)
            x = ReflectionPadding2D(1)(x)
            x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv{}_2".format(i+4))(x)
            x = InstanceNormalization(name="norm{}_2".format(i+4))(x)
            x = Add()([x, x_res])
            x_res = x

        # up-convolution
        for i in range(2):
            x = Conv2DTranspose(channel//2, 3, 2, padding="same", output_padding=1, name="deconv{}_1".format(i+1))(x)
            x = Conv2D(channel//2, (3, 3), strides=1, use_bias=True, padding="same", name="deconv{}_2".format(i+1))(x)
            x = InstanceNormalization(name="norm_deconv"+str(i+1))(x)
            x = Activation("relu")(x)
            channel = channel // 2

        # last block
        x = ReflectionPadding2D(3)(x)
        x = Conv2D(3, (7, 7), strides=1, use_bias=True, padding="valid", name="deconv3")(x)
        x = Activation("tanh")(x)
        
        model = Model(input_img, x, name='Cartoon_Generator')

        return model

    # method for discriminator
    def discriminator(self):
        input_shape=[self.image_size, self.image_size, self.image_channels]
        input_img = Input(shape=input_shape, name="input")

        # first block
        x = Conv2D(32, (3, 3), strides=1, use_bias=True, padding='same', name="conv1")(input_img)
        x = LeakyReLU(alpha=0.2)(x)

        # block loop
        channel = 64
        for i in range(2):
            x = Conv2D(channel, (3, 3), strides=2, use_bias=True, padding='same', name="conv{}_1".format(i+2))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(channel*2, (3, 3), strides=1, use_bias=True, padding='same', name="conv{}_2".format(i+2))(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            channel = channel * 2

        # last block
        x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='same', name="conv4")(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(1, (3, 3), strides=1, use_bias=True, padding='same', activation='sigmoid', name="conv5")(x)

        model = Model(input_img, x, name='Cartoon_Discriminator')

        return model

    # vgg loss function
    def vgg_loss(self, y_true, y_pred):
        # get vgg model
        input_shape=[self.image_size, self.image_size, self.image_channels]
        img_input = Input(shape=input_shape, name="vgg_input")
        vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet')
        vggmodel = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block4_conv4').output)
        x = vggmodel(img_input)
        vgg = Model(img_input, x, name='VGG_for_Feature_Extraction')

        # get l1 loss for the content loss
        y_true = vgg(y_true)
        y_pred = vgg(y_pred)
        content_loss = tf.losses.absolute_difference(y_true, y_pred)

        return content_loss

    # compile each model
    def compile_model(self):
        # init summary writer for tensorboard
        self.callback1 = TensorBoard(self.log_dir+'/discriminator')
        self.callback2 = TensorBoard(self.log_dir+'/generator')
        self.callback3 = TensorBoard(self.log_dir+'/generated_images')
        
        # model stuff
        input_shape=[self.image_size, self.image_size, self.image_channels]
        adam1 = Adam(lr=self.lr)
        adam2 = Adam(lr=self.lr*2)

        # init and add multi-gpu support
        try:
            self.discriminator = multi_gpu_model(self.discriminator(), gpus=self.gpu)
        except:
            self.discriminator = self.discriminator()
        try:
            self.generator = multi_gpu_model(self.generator(), gpus=self.gpu)
        except:
            self.generator = self.generator()

        # compile discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=adam1)

        # compile generator
        input_tensor = Input(shape=input_shape)
        generated_catroon_tensor = self.generator(input_tensor)
        self.discriminator.trainable = False # for here we only train the generator
        discriminator_output = self.discriminator(generated_catroon_tensor)
        self.train_generator = Model(input_tensor, outputs=[generated_catroon_tensor, discriminator_output])
        # add multi-gpu support
        try:
            self.train_generator = multi_gpu_model(self.train_generator, gpus=self.gpu)
        except:
            pass
        self.train_generator.compile(loss=[self.vgg_loss, 'binary_crossentropy'],
                                             loss_weights=[float(self.weight), 1.0],
                                             optimizer=adam2)
        
        # set callback model
        self.callback1.set_model(self.discriminator)
        self.callback2.set_model(self.train_generator)
        self.callback3.set_model(self.train_generator)

    # method for training process
    def train(self):

        # start training
        flip = False
        variance = 1 / 127.5
        start_time = time.time()
        for epoch in range(1, self.epochs+1):

            # create batch generator at each epoch
            batch_generator = DataGenerator(image_size=self.image_size, batch_size=self.batch_size)
            batch_end = len(batch_generator)
            print('Epoch {}'.format(epoch))

            # start training for each batch
            for idx, (photo, cartoon, smooth_cartoon, index) in enumerate(batch_generator):

                # these two tensors measure the output of generator and discriminator
                real = np.ones((self.batch_size,) + (64, 64, 1))
                fake = np.zeros((self.batch_size,) + (64 , 64, 1))

                # check if it is the end of an epoch
                if index + 1 == batch_end:
                    break
                
                # initial training or start training
                if epoch < self.init_epoch:
                    g_loss = self.train_generator.train_on_batch(photo, [photo, real])
                    generated_img = self.generator.predict(photo)
                    print("Batch %d (initial training for generator), g_loss: %.5f, with time: %4.4f" % (idx, g_loss[2], time.time()-start_time))
                    start_time = time.time()
                    write_log(self.callback2, 'g_loss', g_loss[2], idx + (epoch+1)*len(batch_generator))
                    if idx % 20 == 0:
                        write_images(self.callback3, generated_img, 'generated_imgs', idx + (epoch+1)*len(batch_generator))

                    if epoch % 20 == 0 and K.eval(self.train_generator.optimizer.lr) > 0.0001:
                        K.set_value(self.train_generator.optimizer.lr, K.eval(self.train_generator.optimizer.lr)*0.99)

                else:
                    
                    # add noise to the input of discriminator
                    if variance > 0.00001:
                        variance = variance * 0.9999
                        gaussian = np.random.normal(0, variance, (cartoon.shape[1],cartoon.shape[2]))
                        cartoon[:, :, :, 0] = cartoon[:, :, :, 0] + gaussian
                        cartoon[:, :, :, 1] = cartoon[:, :, :, 1] + gaussian
                        cartoon[:, :, :, 2] = cartoon[:, :, :, 2] + gaussian
                        gaussian = np.random.normal(0, variance, (cartoon.shape[1],cartoon.shape[2]))
                        smooth_cartoon[:, :, :, 0] = smooth_cartoon[:, :, :, 0] + gaussian
                        smooth_cartoon[:, :, :, 1] = smooth_cartoon[:, :, :, 1] + gaussian
                        smooth_cartoon[:, :, :, 2] = smooth_cartoon[:, :, :, 2] + gaussian

                    # generate cartoonized images
                    generated_img = self.generator.predict(photo)

                    # to certain probability: flip the label of discriminator
                    if idx % 9 == 0 or np.random.uniform(0, 1) < 0.05:
                        real = fake
                        fake = fake + 1
                        flip = True

                    # train discriminator and adversarial loss
                    real_loss = self.discriminator.train_on_batch(cartoon, real)
                    smooth_loss = self.discriminator.train_on_batch(smooth_cartoon, fake)
                    fake_loss = self.discriminator.train_on_batch(generated_img, fake)
                    d_loss = (real_loss + smooth_loss + fake_loss) / 3

                    # train generator
                    if flip:
                        real = fake
                        fake = fake - 1
                        flip = False

                    g_loss = self.train_generator.train_on_batch(photo, [photo, real])
                    print("Batch %d, d_loss: %.5f, g_loss: %.5f, with time: %4.4f" % (idx, d_loss, g_loss[2], time.time()-start_time))
                    start_time = time.time()

                    # add losses to writer
                    write_log(self.callback1, 'd_loss', d_loss, idx + (epoch+1)*len(batch_generator))
                    write_log(self.callback2, 'g_loss', g_loss[2], idx + (epoch+1)*len(batch_generator))
                    if idx % 20 == 0:
                        write_images(self.callback3, generated_img, 'generated_imgs', idx + (epoch+1)*len(batch_generator))

                    # change learning rate 
                    if epoch % 20 == 0 and K.eval(self.discriminator.optimizer.lr) > 0.0001:
                        K.set_value(self.discriminator.optimizer.lr, K.eval(self.discriminator.optimizer.lr)*0.95)
                    if epoch % 20 == 0 and K.eval(self.train_generator.optimizer.lr) > 0.0001:
                        K.set_value(self.train_generator.optimizer.lr, K.eval(self.train_generator.optimizer.lr)*0.95)
                
                # save model
                if epoch % 50 == 0:
                    self.generator.save_weights(self.model_dir + '/' + 'CartoonGan_generator_epoch_{}.h5'.format(epoch))
                    self.discriminator.save_weigths(self.model_dir + '/' + 'CartoonGan_discriminator_epoch_{}.h5'.format(epoch))
                    self.train_generator.save_weights(self.model_dir + '/' + 'CartoonGan_train_generator_epoch_{}.h5'.format(epoch))
        
        print('Done!')
        self.generator.save('CartoonGan_generator.h5')