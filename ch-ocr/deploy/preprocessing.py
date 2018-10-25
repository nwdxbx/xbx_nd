#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np 
import codecs
from PIL import Image
import time
from PIL import ImageEnhance
import cv2  
import numpy as np  
import math  
  
def stretchImage(data, s=0.005, bins = 2000):    #线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]  
    ht = np.histogram(data, bins);  
    d = np.cumsum(ht[0])/float(data.size)  
    lmin = 0; lmax=bins-1  
    while lmin<bins:  
        if d[lmin]>=s:  
            break  
        lmin+=1  
    while lmax>=0:  
        if d[lmax]<=1-s:  
            break  
        lmax-=1  
    return np.clip((data-ht[1][lmin])/(ht[1][lmax]-ht[1][lmin]), 0,1)  
  
g_para = {}  
def getPara(radius = 5):                        #根据半径计算权重参数矩阵  
    global g_para  
    m = g_para.get(radius, None)  
    if m is not None:  
        return m  
    size = radius*2+1  
    m = np.zeros((size, size))  
    for h in range(-radius, radius+1):  
        for w in range(-radius, radius+1):  
            if h==0 and w==0:  
                continue  
            m[radius+h, radius+w] = 1.0/math.sqrt(h**2+w**2)  
    m /= m.sum()  
    g_para[radius] = m  
    return m  
  
def zmIce(I, ratio=4, radius=300):                     #常规的ACE实现  
    para = getPara(radius)  
    height,width = I.shape  
    zh,zw = [0]*radius + range(height) + [height-1]*radius, [0]*radius + range(width)  + [width -1]*radius  
    Z = I[np.ix_(zh, zw)]  
    res = np.zeros(I.shape)  
    for h in range(radius*2+1):  
        for w in range(radius*2+1):  
            if para[h][w] == 0:  
                continue  
            res += (para[h][w] * np.clip((I-Z[h:h+height, w:w+width])*ratio, -1, 1))  
    return res  
  
def zmIceFast(I, ratio, radius):                #单通道ACE快速增强实现  
    height, width = I.shape[:2]  
    if min(height, width) <=2:  
        return np.zeros(I.shape)+0.5  
    Rs = cv2.resize(I, ((width+1)/2, (height+1)/2))  
    Rf = zmIceFast(Rs, ratio, radius)             #递归调用  
    Rf = cv2.resize(Rf, (width, height))  
    Rs = cv2.resize(Rs, (width, height))  
  
    return Rf+zmIce(I,ratio, radius)-zmIce(Rs,ratio,radius)      
              
def zmIceColor(I, ratio=3, radius=3):               #rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径  
    res = np.zeros(I.shape)  
    for k in range(3):  
        res[:,:,k] = stretchImage(zmIceFast(I[:,:,k], ratio, radius))  
    return res  

if __name__ == '__main__':
    imgpath = "./aaa.jpg"
    path = "/bbb/"
    #for filename in os.listdir(path):
    for i in range(1):
        #imgpath = "/home/robot/real_single_char/102.jpg"
        #imgpath = "/home/robot/Desktop/clear_blur/hu/3.jpg"
        #imgpath = os.path.join(path,filename)
        #if not imgpath[-1] == "g":
        #    continue
        imgpath = "./2.jpg"
        im = Image.open(imgpath)
        b,g,r = im.split()
        im = Image.merge("RGB",(r,g,b))
        #print(imgpath)
        #enh_bri = ImageEnhance.Brightness(im)
        #im = enh_bri.enhance(0.8)

        #enh_con = ImageEnhance.Contrast(im)
        #im = enh_con.enhance(1.3)
        #enh_sha = ImageEnhance.Sharpness(im)
        #im = enh_sha.enhance(1.5)
        img = np.array(im)
        img = zmIceColor(img/255.0)  
        #img = cv2.imread(imgpath,0)
        #img = cv2.resize(img,(299,299))
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.bilateralFilter(img,3,100,100)
        #img = cv2.filter2D(img,-1,kernel)
        #ff,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
       
        #cv2.imshow('img', img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        save_path = imgpath
        cv2.imwrite(save_path,img*255)

