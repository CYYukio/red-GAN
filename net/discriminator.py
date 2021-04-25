import tensorflow as tf
from tensorflow import zeros_initializer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input ,LeakyReLU, BatchNormalization, Dense, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import optimizers


def define_discriminator(img_shape):
    in_src_image = Input(shape=img_shape)
    in_tar_image = Input(shape=img_shape)
    in_img = Concatenate()([in_src_image, in_tar_image])

    conv1 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros_initializer(),
                   activation=None)(in_img)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    # (none, 32, 32, 32)

    conv2 = Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros_initializer(),
                   activation=None)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    # (none, 16, 16, 64)

    conv3 = Conv2D(filters=256, kernel_size=(4, 4), strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros_initializer(),
                   activation=None)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    # (none, 8, 8, 128)

    conv4 = Conv2D(filters=512, kernel_size=(4, 4), strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros_initializer(),
                   activation=None)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(filters=512, kernel_size=(4, 4), strides=2, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros_initializer(),
                   activation=None)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    # (none, 8, 8, 256)

    conv6 = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros_initializer(),
                   activation=None)(conv5)

    # patch_out = Activation('sigmoid')(conv6)  # 5,5

    # define model
    model = Model([in_src_image, in_tar_image], conv6)
    # compile model

    return model


if __name__ == '__main__':
    model = define_discriminator((128, 128, 1))
    opt = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
