# -*- coding:utf-8 -*-

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape
from keras.models import Model,load_model
import pickle
from sklearn.cross_validation import train_test_split
import keras
import os
from PIL import Image
import numpy as np

def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

## 网络结构 ##
ROW = 64
COL = 64
CHANNELS = 3
baseLevel = ROW//2//2
input_img = Input(shape=(ROW,COL,CHANNELS))
x = Conv2D(256, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Flatten()(x)
encoded = Dense(2000)(x)
one_d = Dense(baseLevel*baseLevel*128)(encoded)
fold = Reshape((baseLevel,baseLevel,128))(one_d)

x = UpSampling2D((2, 2))(fold)
x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
amd=keras.optimizers.sgd(lr=0.001)
autoencoder.compile(optimizer=amd, loss='binary_crossentropy')
# 开始导入数据
file = open("./cutImage/cutset.pkl", 'rb')
data, labels = pickle.load(file)
X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.25,random_state=33)

autoencoder.load_weights('./model/autoEncoder_weights.h5', by_name=True)
'''
autoencoder.fit(X_train, X_train,
                nb_epoch=800,
                batch_size=50,
                shuffle=True,
                verbose=2,
                validation_split=0.1)
'''
autoencoder.save_weights('./model/autoEncoder_weights.h5')
# 重建图片
import matplotlib.pyplot as plt
#decoded_imgs = autoencoder.predict(X_test)
print(type(X_test[0]))
print(X_test.shape)
#n = 34
#plt.figure(figsize=(20, 4))

for i in range(15):
    directory="./cutImage/train/"+str(i)+"/"
    print(directory)
    for filename in os.listdir(directory):              #listdir的参数是文件夹的路径
        if filename.endswith('png'):
            img = Image.open(directory+filename)
            resize_mode = Image.ANTIALIAS
            img = img.resize((64, 64), resize_mode)
            img = pil_to_nparray(img)
            img /= 255.
            img = np.array([img])
            print(img.shape)
            decoded_imgs = autoencoder.predict(img)
            # 画原始图片
            ax = plt.subplot(2, 1, 1)
            plt.imshow(img.reshape(64, 64, 3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            # 画重建图片
            ax = plt.subplot(2, 1, 2)
            plt.imshow(decoded_imgs.reshape(64, 64, 3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()
'''
for i in range(n):
    k = i+1
    # 画原始图片
    ax = plt.subplot(2, n, k)
    plt.imshow(X_test[k].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    # 画重建图片
    ax = plt.subplot(2, n, k + n)
    plt.imshow(decoded_imgs[k].reshape(64, 64,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 编码得到的特征
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    k = i + 1
    ax = plt.subplot(1, n, k)
    plt.imshow(encoded[k].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''