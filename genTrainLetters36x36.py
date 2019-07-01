from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import os
import cv2
import skimage
import matplotlib as cm
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import scipy.misc
import imageio
from datetime import datetime
import shutil

train_root = "./train/"
test_root = "./test/"

train_file = 'train.txt'
test_file = 'test.txt'

#train_num = 168168
#train_num = 168
train_num = 16688

#test_num = 16888
#test_num = 16
test_num = 168

fontMax = 36
fontMin = 28
shakeAng = 6
letterMax = 35    #35
letterMin = 0    #0

# 随机字母:
def rndNumIndex():
    num = random.randint(0,35)
    if (num <10):
       num1 = num+48
    else:
       num1 = num+65-10
    return num1

def rndNumIndex2(num):
    if (num <10):
       num1 = num+48
    else:
       num1 = num+65-10
    return num1

def rndAngle(x):
    return random.randint(-x,x)

# 随机颜色1:
def rndColor():
    return (random.randint(0, 68), random.randint(0, 68), random.randint(0, 68))

# 随机颜色2:
def rndColor2():
    return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def plotnoise(img, mode):
    #if mode is not None:
    gimg = skimage.util.random_noise(img, mode=mode)
    #else:
    #    gimg = img
    return gimg
    
def rndPos(x):
    return random.randint(0,x)

def rndFontSize(x,y):
    return random.randint(x,y)

def getChar(mode):
    width = 36
    height = 36
    image = Image.new('RGB', (width, height), (255, 255, 255))
    # 创建Font对象:
    fontSize = rndFontSize(fontMin,fontMax)
    font = ImageFont.truetype('Harrington.ttf', fontSize)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 填充每个像素:
    for x in range(width):
       for y in range(height):
           draw.point((x, y), fill=rndColor())
    # 输出文字:
    charNum = random.randint(letterMin,letterMax)
    char =  chr(rndNumIndex2(charNum))
    #label =  str(charNum)
    
    for t in range(1):
        #draw.text((28 * t + 8 , 5), char, font=font, fill=rndColor2())
        draw.text((rndPos(36-fontSize+6) , rndPos(36-fontSize)), char, font=font, fill=rndColor2())        

    # 模糊:
    #image = image.filter(ImageFilter.BLUR)

    gray = image.convert('L')
    image1 = gray.rotate( rndAngle(shakeAng), Image.BILINEAR )

    image2 = np.array(image1)
    image3 = plotnoise(image2, 'gaussian') 

    dateStr = datetime.utcnow().strftime('%m%d%H%M%S%f')[:-3]
    fileStr = dateStr +'_' + str(charNum) + '.jpg'
    imagePath1 = train_root + fileStr
    imagePath2 = test_root + fileStr

    if (mode == 0):
        imageio.imwrite(imagePath1, image3)

    if (mode == 1):
       imageio.imwrite(imagePath2, image3)

    print ('gen char = ' + char)

    if (mode == 0):
        img_path = train_root + fileStr   
        charLine = img_path + ' ' + str(charNum)

    if (mode == 1): 
        img_path = test_root + fileStr   
        charLine = img_path + ' ' + str(charNum)

    return charLine

def mkdir(name):
    try:
       # Create target Directory
       os.mkdir(name)
       print("Directory " , name ,  " Created ") 
    except FileExistsError:
       print("Directory " , name ,  " already exists")

mkdir (train_root)
mkdir (test_root)

f = open(train_file, 'a+')
ftest = open(test_file, 'a+')

for train_i in range(train_num):
    f.write(getChar(0)+ '\n')

for train_i in range(test_num):
    print('count ='+ str(train_i))
    ftest.write(getChar(1)+ '\n')
    print('end count \n') 

f.close()
ftest.close()


