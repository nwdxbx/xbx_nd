#! /usr/env/bin python
# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
import random

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance
from skimage import draw as drawcurve

def gaussianNoisy(im):
    for _i in range(len(im)):
        im[_i] += random.gauss(0.2, 0.3)
    return im

def gene_img(text):
    
    im = Image.open("../background_img/background_"+str(random.randint(1,18))+".jpg")
    
    im = im.resize((35*8,40))
    
    sdsds = random.uniform(-2,2)
    im = im.rotate(sdsds, Image.BICUBIC)
   
    img = np.asarray(im)
    img.flags.writeable = True
    img_r = gaussianNoisy(img[:, :, 0].flatten())+random.randint(-50,80)
    img_g = gaussianNoisy(img[:, :, 1].flatten())+random.randint(-50,80)
    img_b = gaussianNoisy(img[:, :, 2].flatten())+random.randint(-50,80)
    img[:, :, 0] = img_r.reshape([40,35*8])
    img[:, :, 1] = img_g.reshape([40,35*8]) 
    img[:, :, 2] = img_b.reshape([40,35*8])
    img =  Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(im)
    char_list = list(text.strip())
    i , j = 0, 0
    start_x, start_y = random.randint(5,25),random.randint(4,6)
    font_size = random.randint(21,23)
    stride_x, stride_y  = font_size+2, font_size
    font_path = "../fonts/1.ttf"
    font = ImageFont.truetype(font_path, font_size)
    for idx,char in enumerate(char_list):
        x_min = start_x + i * stride_x
        y_min = start_y + j * stride_y       
        draw.text(( x_min , y_min ), char,(random.randint(0,30),random.randint(0,30),random.randint(0,30)) ,font=font)
        width , height = draw.textsize(char, font = font)
        x_max = x_min + width
        y_max = y_min + height
        i = i + 1
    del draw

    enh_col = ImageEnhance.Color(im)
    im = enh_col.enhance(random.uniform(0.5,1.5))

    enh_bri = ImageEnhance.Brightness(im)
    im = enh_bri.enhance(random.uniform(0.9,1.3)) 
   
    enh_con = ImageEnhance.Contrast(im)
    im = enh_con.enhance(random.uniform(0.8,1.2))
    
    enh_sha = ImageEnhance.Sharpness(im)
    im = enh_sha.enhance(random.uniform(0.8,1.5))

    # change to the opencv(numpy) image data type
    cv_im = np.array(im)
    '''
    if random.randint(0,8) == 8:
        for iii in range(random.randint(3,5)):
            rr, cc=drawcurve.bezier_curve(random.randint(0,40),random.randint(225,235),random.randint(0,40),random.randint(235,245),random.randint(0,40),random.randint(245,255),random.randint(3,5))
            drawcurve.set_color(cv_im,[rr,cc],[random.randint(0,255),random.randint(0,255),random.randint(0,255)])
    '''


    # enlarge the image and use erode to widen the stride
    factor = random.randint(4,6)
    new_shape = (cv_im.shape[1] * factor, cv_im.shape[0] * factor)
    im_enlarge = cv2.resize(cv_im,new_shape)
    k = random.choice([1,3])
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k))
    im_enlarge = cv2.erode(im_enlarge, kernel, iterations = 1)
    result_im = cv2.resize(im_enlarge, (cv_im.shape[1], cv_im.shape[0]))

    w = result_im.shape[1]
    h = result_im.shape[0]
    for i in range(int(w)):
        ww = random.randint(0,w-1)
        hh = random.randint(0,h-1)
        #noise = [[0,0,0],[255,255,255]]
        noise = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        result_im [hh,ww] = random.choice(noise)

    # use Gaussion filter to blur the img
    k = random.choice([[3,1],[5,1],[7,1]])
    result_im = cv2.GaussianBlur(result_im ,( k[0], k[0] ),k[1])
    blur_im = result_im
    
    #blur_im = cv2.resize(blur_im, (random.randint(60,70),random.randint(60,70)))
    im = Image.fromarray(blur_im).rotate(-sdsds, Image.BICUBIC)
    im = im.crop((4,4,260,36))
    cv_im = np.array(im)
    
    #print(cv_im.shape)
    return cv_im
