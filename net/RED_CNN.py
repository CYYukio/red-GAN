import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Input ,LeakyReLU, BatchNormalization, Dense, Concatenate, Add
from tensorflow.keras.initializers import RandomNormal, zeros
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def redcnn(img_shape=(64, 64, 1), kernel_size=(3, 3), filter_size=96, stride=1):
    in_image = Input(shape=img_shape)
    # encoder_layer
    conv1 = Conv2D(filters=filter_size,
                   kernel_size=kernel_size,
                   strides=stride,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros(),
                   activation=None)(in_image)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters=filter_size,
                   kernel_size=kernel_size,
                   strides=stride,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros(),
                   activation=None)(conv1)
    conv2 = shortcut_deconv8 = Activation('relu')(conv2)

    conv3 = Conv2D(filters=filter_size,
                   kernel_size=kernel_size,
                   strides=stride,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros(),
                   activation=None)(conv2)
    conv3 = Activation('relu')(conv3)

    conv4 = Conv2D(filters=filter_size,
                   kernel_size=kernel_size,
                   strides=stride,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros(),
                   activation=None)(conv3)
    conv4 = shortcut_deconv6 = Activation('relu')(conv4)

    conv5 = Conv2D(filters=filter_size,
                   kernel_size=kernel_size,
                   strides=stride,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.01),
                   bias_initializer=zeros(),
                   activation=None)(conv3)
    conv5 = Activation('relu')(conv4)

    # decoder layer
    deconv6 = Conv2DTranspose(filters=filter_size,
                              kernel_size=kernel_size,
                              strides=stride,
                              padding='same',
                              kernel_initializer=RandomNormal(stddev=0.01),
                              bias_initializer=zeros())(conv5)
    # deconv6 += shortcut_deconv6
    deconv6 = Add()([deconv6, shortcut_deconv6])
    deconv6 = Activation('relu')(deconv6)

    deconv7 = Conv2DTranspose(filters=filter_size,
                              kernel_size=kernel_size,
                              strides=stride,
                              padding='same',
                              kernel_initializer=RandomNormal(stddev=0.01),
                              bias_initializer=zeros())(deconv6)
    deconv7 = Activation('relu')(deconv7)

    deconv8 = Conv2DTranspose(filters=filter_size,
                              kernel_size=kernel_size,
                              strides=stride, padding='same',
                              kernel_initializer=RandomNormal(stddev=0.01),
                              bias_initializer=zeros())(deconv7)
    # deconv8 += shortcut_deconv8
    deconv8 = Add()([deconv8, shortcut_deconv8])
    deconv8 = Activation('relu')(deconv8)

    deconv9 = Conv2DTranspose(filters=filter_size,
                              kernel_size=kernel_size,
                              strides=stride, padding='same',
                              kernel_initializer=RandomNormal(stddev=0.01),
                              bias_initializer=zeros())(deconv8)
    deconv9 = Activation('relu')(deconv9)

    deconv10 = Conv2DTranspose(filters=1,
                               kernel_size=kernel_size,
                               strides=stride, padding='same',
                               kernel_initializer=RandomNormal(stddev=0.01),
                               bias_initializer=zeros())(deconv9)
    # deconv10 += in_image
    deconv10 = Add()([deconv10, in_image])
    out_image = Activation('relu')(deconv10)

    model = Model(in_image, out_image)

    return model


if __name__ == '__main__':
    redcnn().summary()