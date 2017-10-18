import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def draw(filename,result):
    img = Image.open(filename)
    w,h=img.size
    print(w,h)
    draw = ImageDraw.Draw(img)
    result=np.array(result)
    x=result[0][0]
    y=result[0][1]
    x=w/2
    y=h/2
    print(x,y)
    angle=result[0][2]
    height=result[0][3]
    width=result[0][4]

    anglePi = -angle*math.pi/180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x1=x-0.5*width
    y1=y-0.5*height

    x0=x+0.5*width
    y0=y1

    x2=x1
    y2=y+0.5*height

    x3=x0
    y3=y2

    x0n= (x0 -x)*cosA -(y0 - y)*sinA + x
    y0n = (x0-x)*sinA + (y0 - y)*cosA + y

    x1n= (x1 -x)*cosA -(y1 - y)*sinA + x
    y1n = (x1-x)*sinA + (y1 - y)*cosA + y

    x2n= (x2 -x)*cosA -(y2 - y)*sinA + x
    y2n = (x2-x)*sinA + (y2 - y)*cosA + y

    x3n= (x3 -x)*cosA -(y3 - y)*sinA + x
    y3n = (x3-x)*sinA + (y3 - y)*cosA + y


    draw.line([(x0n, y0n),(x1n, y1n)], fill=(0, 0, 255))
    draw.line([(x1n, y1n),(x2n, y2n)], fill=(255, 0, 0))
    draw.line([(x2n, y2n),(x3n, y3n)],fill= (0,0,255))
    draw.line([(x0n, y0n), (x3n, y3n)],fill=(255,0,0))

    plt.imshow(img)
    plt.show()

def draw2(filename, result,angle,y):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    result = np.array(result)
    x = result[0][0]  # 提取x
    height = result[0][1]  # 提取高度
    width = result[0][2]  # 提取宽度
    height=height+15
    angle=angle+10

    anglePi = -angle * math.pi / 180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    draw.line([(x0n, y0n), (x1n, y1n)], fill=(0, 0, 255))
    draw.line([(x1n, y1n), (x2n, y2n)], fill=(255, 0, 0))
    draw.line([(x2n, y2n), (x3n, y3n)], fill=(0, 0, 255))
    draw.line([(x0n, y0n), (x3n, y3n)], fill=(255, 0, 0))

    plt.imshow(img)
    plt.show()

file = '/home/robot/cy/grasp/cutImage/train/5/pcd0470r.png'
file2 = '/home/robot/cy/grasp/trainImage/jpg/5/pcd0470r.png'


result=[[ 53.308,57.795,0,33.838,57]]
draw(file,result)
#result=[[ 28.5+297.128,53.5+117.749,71.03,60.284,33.838]]
#draw(file2,result)