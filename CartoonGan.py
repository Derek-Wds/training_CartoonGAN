import time
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import TensorBoard
from utils import *

# class for CartoonGAN
class CartoonGAN():
    def __init__(self, args):
        self.model_name = 'CartoonGAN'
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.log_dir = args.log_dir
        self.lr = args.lr
        self.weight = args.weight
    
    # method for generator
    def generator(self):
        input_shape=[self.image_size, self.image_size, self.image_channels]
        input_img = Input(shape=input_shape, name="input")

        # first block
        x = Lambda(lambda t: tf.pad(t, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT'))(input_img)
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
            x = Lambda(lambda t: tf.pad(t, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))(x)
            x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv{}_1".format(i+4))(x)
            x = InstanceNormalization(name="norm{}_1".format(i+4))(x)
            x = Activation("relu")(x)
            x = Lambda(lambda t: tf.pad(t, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))(x)
            x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv{}_2".format(i+4))(x)
            x = InstanceNormalization(name="norm{}_2".format(i+4))(x)
            x = Add()([x, x_res])
            x_res = x

        # up-convolution
        for i in range(2):
            x = Conv2DTranspose(channel//2, 3, 2, padding="same", name="deconv{}_1".format(i+1))(x)
            x = Conv2D(channel//2, (3, 3), strides=1, use_bias=True, padding="same", name="deconv{}_2".format(i+1))(x)
            x = InstanceNormalization(name="norm_deconv"+str(i+1))(x)
            x = Activation("relu")(x)
            channel = channel // 2

        # last block
        x = Lambda(lambda t: tf.pad(t, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT'))(x)
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
        content_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        return content_loss

    # compile each model
    def compile_model(self):
        # init summary writer for tensorboard
        self.callback1 = TensorBoard(self.log_dir+'_discriminator')
        self.callback2 = TensorBoard(self.log_dir+'_generator')
        
        # model stuff
        input_shape=[self.image_size, self.image_size, self.image_channels]
        adam1 = Adam(lr=self.lr)
        adam2 = Adam(lr=self.lr)

        # init
        self.discriminator = self.discriminator()
        self.generator = self.generator()

        # compile discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=adam1)

        # compile generator
        input_tensor = Input(shape=input_shape)
        generated_catroon_tensor = self.generator(input_tensor)
        discriminator_output = self.discriminator(generated_catroon_tensor)
        self.train_generator = Model(input_tensor, outputs=[generated_catroon_tensor, discriminator_output])
        self.train_generator.compile(loss=[self.vgg_loss, 'binary_crossentropy'],
                                             loss_weights=[float(self.weight), 1.0],
                                             optimizer=adam2)
        
        # set callback model
        self.callback1.set_model(self.discriminator)
        self.callback2.set_model(self.train_generator)

    # method for training process
    def train(self, batch_generator):
        # these two tensors measure the output of generator and discriminator
        real = np.ones((batch_generator.batch_size,) + (64, 64, 1))
        fake = np.zeros((batch_generator.batch_size,) + (64 , 64, 1))

        # start training
        start_time = time.time()
        for i in range(self.epochs):
            print('Epoch {}'.format(i+1))
            for idx, (cartoon, smooth_cartoon, photo) in enumerate(batch_generator):
                print('hh')
                print(idx)
                # train discriminator
                generated_img = self.generator.predict(photo)
                real = self.discriminator.train_on_batch(cartoon, real)
                smooth = self.discriminator.train_on_batch(smooth_cartoon, fake)
                fake = self.discriminator.train_on_batch(generated_img, fake)
                d_loss = (real + smooth + fake) / 3

                # train generator
                g_loss = self.train_generator.train_on_batch(photo,[photo, real])
                print("Batch %d, d_loss: %.5f, g_loss: %.5f, with time: %4.4f" % (idx, d_loss, g_loss, time.time()-start_time))

                # add losses to writer
                write_log(self.callback1, 'd_loss', d_loss, idx)
                write_log(self.callback2, 'g_loss', g_loss, idx)

                # change learning rate 
                if idx % 500 == 0 and self.discriminator.optimizer.lr > 0.0001:
                    K.set_value(self.discriminator.optimizer.lr, self.discriminator.optimizer.lr*0.95)
                    K.set_value(self.train_generator.optimizer.lr, self.train_generator.optimizer.lr*0.95)
        
        print('Done!')
        self.generator.save('CartoonGan_generator.h5')