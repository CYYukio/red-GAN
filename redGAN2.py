from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from numpy import load, expand_dims, concatenate
import numpy as np
from keras.applications.vgg16 import VGG16
from net.utils import get_random_patchs
from net.discriminator import define_discriminator
from net.RED_CNN import redcnn
import tensorflow as tf


def feature_loss(y_true, y_pred):
    h = y_true.shape[1]
    w = y_true.shape[2]
    c = y_true.shape[3]

    return tf.norm(y_true-y_pred) / (h*w*c)

'''
def style_loss(y_true, y_pred):
    h = y_true.shape[1]
    w = y_true.shape[2]
    c = y_true.shape[3]

    y_true_flat = tf.reshape(y_true, (h*w, c))
    y_pred_flat = tf.reshape(y_pred, (h*w, c))

    true = tf.matmul(tf.transpose(y_true_flat), y_true_flat) / (h*w*c)
    pred = tf.matmul(tf.transpose(y_pred_flat), y_pred_flat) / (h*w*c)

    return tf.square(true - pred)
'''

def style_loss(y_true, y_pred):
    h = y_true.shape[1]
    w = y_true.shape[2]
    c = y_true.shape[3]

    y_true_flat = tf.reshape(y_true, (h*w, c))
    y_pred_flat = tf.reshape(y_pred, (h*w, c))

    Gram_true = tf.matmul(tf.transpose(y_true_flat), y_true_flat) / (h*w*c)
    Gram_pred = tf.matmul(tf.transpose(y_pred_flat), y_pred_flat) / (h*w*c)

    return tf.sqrt(tf.norm(Gram_true - Gram_pred))


class redGAN():
    def __init__(self):
        # part1 定义基本参数 ##################
        self.epochs = 20
        self.img_size = 128
        self.img_shape = (128, 128, 1)

        self.pool_size = 1000

        self.d_outsize = 4  # 判别器输出为16*16
        self.opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

        self.LDCT_list_all, self.NDCT_list_all = self.loaddata()
        self.ldctpatch_pool, self.ndctpatch_pool = self.update_pool()
        ######################################

        # part2 定义网络模型 ##################
        self.discriminator = define_discriminator((128, 128, 1))  # 要做concate
        self.generator = redcnn((128, 128, 1))
        # self.generator.load_weights('redGAN_generator_epoch1000.h5')
        # self.discriminator.load_weights('redGAN_discrminator_epoch1000.h5')

        self.redGAN = self.GAN()

        self.discriminator.compile(loss='mse', optimizer=self.opt)
        self.generator.compile(loss='mse', optimizer=self.opt)
        self.redGAN.compile(loss=['mse', 'mse'], loss_weights=[1, 100])  # 感知损失暂时不用

        self.VGG16 = VGG16(weights='imagenet', include_top=False)
        self.perceptual_Feature = self.Feature()
        self.perceptual_Style = self.Style()
        self.p_model = self.perceptual_model()
        self.p_model.compile(loss=[feature_loss, style_loss])
        ######################################

    def loaddata(self):
        dataset1 = load('./data/LDCT&NDCT_large1.npz')

        LDCT_list1, NDCT_list1 = dataset1['arr_0'], dataset1['arr_1']

        print('load:', LDCT_list1.shape, NDCT_list1.shape)
        return LDCT_list1, NDCT_list1

    def update_pool(self):
        idx1 = np.random.randint(0, self.LDCT_list_all.shape[0], self.pool_size)  # 取1000个图

        LDCT_list1000, NDCT_list1000 = self.LDCT_list_all[idx1], self.NDCT_list_all[idx1]
        print('>>get ', LDCT_list1000.shape[0], ' img')

        LDCT_patchs1000, NDCT_patchs1000 = get_random_patchs(LDCT_list1000, NDCT_list1000, self.img_size)

        return LDCT_patchs1000, NDCT_patchs1000

    def GAN(self):
        self.discriminator.trainable = False

        src_in = Input(self.img_shape)

        gen_out = self.generator(src_in)

        dis_out = self.discriminator([src_in, gen_out])

        model = Model(src_in, [gen_out, dis_out])

        return model

    def Feature(self):
        self.VGG16.trainable = False

        VGG_feature = Model(inputs=self.VGG16.input, outputs=self.VGG16.get_layer("block3_conv3").output)
        return VGG_feature

    def Style(self):
        self.VGG16.trainable = False

        VGG_style = Model(inputs=self.VGG16.input, outputs=self.VGG16.get_layer("block1_conv2").output)

        return VGG_style

    def perceptual_model(self):
        self.VGG16.trainable = False
        src_in = Input(self.img_shape)

        gen_out = self.generator(src_in)

        VGG_in_fake = Concatenate()([gen_out, gen_out, gen_out])

        feature_out = self.perceptual_Feature(VGG_in_fake)
        style_out = self.perceptual_Style(VGG_in_fake)

        model = Model(src_in, [feature_out, style_out])
        return model

    def train(self):
        realLabel = np.ones((1, self.d_outsize, self.d_outsize, 1))
        fakeLabel = np.zeros((1, self.d_outsize, self.d_outsize, 1))

        for epoch in range(self.epochs):
            if epoch != 0:
                self.ldctpatch_pool, self.ndctpatch_pool = self.update_pool()
            for roundi in range(1000):
                trainX = self.ldctpatch_pool[roundi]
                trainY = self.ndctpatch_pool[roundi]

                trainX = expand_dims(trainX, axis=-1)  # 网络输入是三通道
                trainY = expand_dims(trainY, axis=-1)
                trainX = expand_dims(trainX, axis=0)
                trainY = expand_dims(trainY, axis=0)
                trainX = trainX.astype(np.float64)
                trainY = trainY.astype(np.float64)

                fakeX = self.generator.predict(trainX)

                d_loss1 = self.discriminator.train_on_batch([trainX, trainY], realLabel)
                d_loss2 = self.discriminator.train_on_batch([trainX, fakeX], fakeLabel)

                g_loss, gan_g, gan_d = self.redGAN.train_on_batch(trainX, [trainY, realLabel])  # 训练方向

                # 感知损失 #############################
                trainY = concatenate([trainY, trainY, trainY], axis=-1)
                feature_valid = self.perceptual_Feature(trainY)
                sytle_valid = self.perceptual_Style(trainY)

                p_loss, f_loss, s_loss1 = self.p_model.train_on_batch(trainX,
                                                                               [feature_valid,
                                                                                sytle_valid])
                ########################################

                if roundi % 100 == 0:
                    print('%d>%d, d1[%.3f] d2[%.3f] | g[%.3f] [%.3f, %.3f]' % (
                        epoch, roundi + 1, d_loss1, d_loss2, g_loss, gan_g, gan_d))

                    print('    perceptual [%.3f] | feature [%.3f] style [%.3f]' % (
                        p_loss, f_loss, s_loss1))

        filename1 = 'redGAN_generator_epoch40.h5'
        filename2 = 'redGAN_discrminator_epoch40.h5'
        self.generator.save(filename1)
        self.discriminator.save(filename2)


if __name__ == '__main__':
    gan = redGAN()
    gan.train()

