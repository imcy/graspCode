from __future__ import division, print_function, absolute_import
from PIL import Image
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import pandas as pd
import os
from tflearn.layers.estimator import regression


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


#network = fully_connected(network, 11, activation='softmax')
#network = regression(network, optimizer='rmsprop',
#                     loss='categorical_crossentropy',
#                     learning_rate=0.0001)
# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)
model.load("./model/modelvgg.tfl")
pd_data = pd.DataFrame()
filelist=[]
for i in range(15):
    directory="./testImage/"+str(i)+"/"
    print(directory)
    for filename in os.listdir(directory):              #listdir的参数是文件夹的路径
        if filename.endswith('png'):
            print(filename[-8:-5])
            filelist.append(filename[-8:-5])
            img = Image.open(directory+filename)
            resize_mode = Image.ANTIALIAS
            img = img.resize((224, 224), resize_mode)
            img = np.array([np.array(img)])
            y = model.predict(img)
            pd_data = pd_data.append(pd.DataFrame(y))
            print(y)                                  #此时的filename是文件夹中文件的名称
pd_data.to_csv('./feature/testfv.csv')
filelist=pd.DataFrame(filelist)
filelist.to_csv('./feature/testfilename.csv')
