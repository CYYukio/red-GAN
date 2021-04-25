import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from net.RED_CNN import redcnn
from numpy import load
from net.utils import get_random_patchs
from numpy import expand_dims
from net.discriminator2 import define_discriminator
import keras


def update_pool(LDCT_list, NDCT_list):
    LDCT_patchs1000, NDCT_patchs1000 = get_random_patchs(LDCT_list, NDCT_list, 128)
    print('>>update  pool')
    return LDCT_patchs1000, NDCT_patchs1000


def train(epochs):
    discriminatorA = define_discriminator((128, 128, 1))
    discriminatorB = define_discriminator((128, 128, 1))
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    discriminatorA.compile(loss='mse', optimizer=opt)
    discriminatorB.compile(loss='mse', optimizer=opt)

    dataset = load('../data/LDCT&NDCT_large1.npz')
    LDCT_list, NDCT_list = dataset['arr_0'], dataset['arr_1']

    steps = 1000 * epochs
    LDCT, NDCT = update_pool(LDCT_list, NDCT_list)

    realLabel = np.ones((1, 16, 16, 1))
    fakeLabel = np.zeros((1, 16, 16, 1))

    for i in range(steps):
        if (i+1)%1000==0:
            LDCT, NDCT = update_pool(LDCT_list, NDCT_list)
        LDCT_patch, NDCT_patch = LDCT[i % 1000], NDCT[i % 1000]
        trainX = expand_dims(LDCT_patch, axis=-1)  # 网络输入是三通道
        trainY = expand_dims(NDCT_patch, axis=-1)

        trainX = expand_dims(trainX, axis=0)
        trainY = expand_dims(trainY, axis=0)
        trainX = trainX.astype(np.float64)
        trainY = trainY.astype(np.float64)

        lossA_real = discriminatorA.train_on_batch(trainX, realLabel)
        lossA_fake = discriminatorA.train_on_batch(trainY, fakeLabel)

        lossB_real = discriminatorB.train_on_batch(trainY, realLabel)
        lossB_fake = discriminatorB.train_on_batch(trainX, fakeLabel)

        if i % 100 == 0:
            print('%d> lossA[%3.f, %3.f] lossB[%.3f,%.3f]' % (
                i, lossA_real, lossA_fake, lossB_real, lossB_fake))

    filename = 'disA.h5'
    discriminatorA.save(filename)
    filename = 'disB.h5'
    discriminatorB.save(filename)


if __name__ == '__main__':
    train(10)
