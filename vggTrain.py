from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pandas as pd
import numpy as np
# Data loading and preprocessing
import pickle

file = open("./trainImage/dataset.pkl",'rb')
testfile=open('./testImage/testset.pkl','rb')
data, labels = pickle.load(file)
testdata, testlabels = pickle.load(testfile)

# Building 'VGG Network'
network = input_data(shape=[None, 224, 224, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 15, activation='softmax')
network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)


# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                   max_checkpoints=1, tensorboard_verbose=0)

model.load("./model/modelvgg.tfl")

model.fit(data, labels, n_epoch=50, shuffle=True,
          show_metric=True, batch_size=32,snapshot_step=100,
          snapshot_epoch=False, run_id='vgg_train',validation_set=(testdata,testlabels))
model.save("./model/modelvgg.tfl")