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
    
    # background image cropped from the user id image
    #im = Image.open("/home/robot/hcl/generate_OCRimages/background_img/background_"+str(random.randint(1,6))+".jpg")
    im = Image.open("../background_img/background_"+str(random.randint(1,18))+".jpg")
    font_size = random.randint(21,25)
    stride_x = font_size-8
    im = im.resize((265,50))
    sdsds = random.uniform(-3,3)
    im = im.rotate(sdsds, Image.BICUBIC)
    draw = ImageDraw.Draw(im)
    char_list = list(text.strip())
    i , j = 0, 0
    start_x, start_y = random.randint(11,14),random.randint(9,12)
    font_path = "../fonts/idnum.ttf"
    font = ImageFont.truetype(font_path, font_size)   
    for idx,char in enumerate(char_list):
        x_min = start_x + idx * stride_x+random.randint(-2,1)
        draw.text(( x_min , start_y ), char,(random.randint(0,30),random.randint(0,30),random.randint(0,30)) ,font=font)
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
    # enlarge the image and use erode to widen the stride
    '''
    factor = random.randint(2,4)
    print(factor)
    new_shape = (cv_im.shape[1] * factor, cv_im.shape[0] * factor)
    im_enlarge = cv2.resize(cv_im,new_shape)
    k = random.choice([1,1,3])
    print(k)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k))
    im_enlarge = cv2.erode(im_enlarge, kernel, iterations = 1)
    result_im = cv2.resize(im_enlarge, (cv_im.shape[1], cv_im.shape[0]))
    '''
    result_im = cv_im
    w = result_im.shape[1]
    h = result_im.shape[0]
    for i in range(int(w)):
        ww = random.randint(0,w-1)
        hh = random.randint(0,h-1)
        #noise = [[0,0,0],[255,255,255]]
        noise = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        result_im [hh,ww] = random.choice(noise)

    # use Gaussion filter to blur the img
    #k = random.choice([[5,1],[7,1]])
    #print(k)
    #result_im = cv2.GaussianBlur(result_im ,( k[0], k[0] ),k[1])
    blur_im = result_im
    
    #blur_im = cv2.resize(blur_im, (random.randint(60,70),random.randint(60,70)))
    im = Image.fromarray(blur_im).rotate(-sdsds, Image.BICUBIC)
    im = im.crop((4,7,260,39))
    cv_im = np.array(im)
    return cv_im

