# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from sklearn.externals import joblib
from PIL import Image
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from drawPic import draw,draw2
import random
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import os
import pandas as pd


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
# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)

model.load("./model/modelvgg.tfl") #提取图片特征

predict_all = joblib.load("./train_model-all.m") #回归模型
predict_angle = joblib.load("./train_model-all_angle.m") #回归模型
predict_y = joblib.load("./train_model-y.m") #回归模型
#svc = joblib.load("./train_model-s.m")
'''
filename='/home/robot/cy/grasp/trainImage/jpg/10/pcd0685r.png'
img = Image.open(filename)
resize_mode=Image.ANTIALIAS
img = img.resize((224, 224), resize_mode)
img=np.array([np.array(img)])

y=model.predict(img)
'''

for i in range(0,15):
    #directory = "./test/"
    directory="./testImage/"+str(i)+"/"
    print(directory)
    for filename in os.listdir(directory):              #listdir的参数是文件夹的路径
        if filename.endswith('png'):
            print(filename)
            img = Image.open(directory+filename)
            resize_mode = Image.ANTIALIAS
            img = img.resize((224, 224), resize_mode)
            img = np.array([np.array(img)])
            y = model.predict(img)
            temp = y
            y_axis = predict_y.predict(y)  # 得到坐标y
            #y = np.c_[y, y_axis]
            #angle = predict_angle.predict(y)  # 得到角度值
            #y=np.c_[y,angle]

            result = predict_all.predict(y) # 得到其他所有值

            x = result[0][0]  # 提取x
            height = result[0][1]  # 提取高度
            width = result[0][2]  # 提取宽度
            data = np.c_[y, y_axis,x,height, width]
            angle = predict_angle.predict(data)

            #for i in range(1, 18):
                #print(angle)
                #data = np.c_[temp, x, y_axis, angle, height, width]
                #res = svc.predict(data)
                #print(res)
                #if res[0]==1:
                #    break
                #angle = angle + 10
                #if angle > 180:
                 #   angle = angle - 180

            draw2(directory+filename, result, angle, y_axis)

