import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from net.RED_CNN import redcnn
from numpy import load
from net.utils import get_random_patchs
from numpy import expand_dims
import os
import keras

def update_pool(LDCT_list, NDCT_list):
    print('>>get ', 1000, ' img')

    LDCT_patchs1000, NDCT_patchs1000 = get_random_patchs(LDCT_list, NDCT_list, 128)

    return LDCT_patchs1000, NDCT_patchs1000


def train(epochs):
    red = redcnn((128, 128, 1))
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    red.compile(loss='mse', optimizer=opt)
    dataset = load('../data/LDCT&NDCT_large1.npz')
    LDCT_list, NDCT_list = dataset['arr_0'], dataset['arr_1']

    steps = 1000 * epochs
    LDCT, NDCT = update_pool(LDCT_list, NDCT_list)
    for i in range(steps):
        if (i+1)%1000==0:
            LDCT, NDCT = update_pool(LDCT_list, NDCT_list)
        LDCT_patch, NDCT_patch = LDCT[i % 1000], NDCT[i % 1000]
        trainX = expand_dims(LDCT_patch, axis=-1)  # 网络输入是三通道
        trainY = expand_dims(NDCT_patch, axis=-1)

        trainX = expand_dims(trainX, axis=0)
        trainY = expand_dims(trainY, axis=0)

        loss = red.train_on_batch(trainX, trainY)

        if i % 100 == 0:
            print('>', i + 1, ', loss ', loss)

    filename = 'red_epoch10.h5'
    red.save(filename)


if __name__ == '__main__':
    train(10)
