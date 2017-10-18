## 抓取数据集的vggNet训练方法
先对数据集进行类别划分，再用tflearn搭建的vggNet对数据进行训练，得到分类模型，取特征数据，并对特征数据进行输出标签长度，宽度，角度，x,y坐标的回归。

### ConAutoEncode
剪裁图片的卷积自编码器实现
### VggTrain
用vggNet训练抓取数据集
### drawPic
在图片中根据长度，宽度，角度，x,y坐标画出相对应的矩形框
### getfeature
根据训练好的Vgg模型得到图片特征
### image2pkl
将图片打包成pkl格式
### Regressor
根据提取的图片特征进行回归
