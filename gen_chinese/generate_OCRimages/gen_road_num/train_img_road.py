#! /usr/env/bin python
# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
import random
import textwrap
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance

def gaussianNoisy(im):
    for _i in range(len(im)):
        im[_i] += random.gauss(0.2, 0.3)
    return im

def gene_img(text):
    # background image cropped from the user id image
    im = Image.open("../background_img/background_"+str(random.randint(1,18))+".jpg")
    #im = Image.open("/home/robot/hcl/generate_OCRimages/background_img/background_"+str(random.randint(1,18))+".jpg")
    w = 265
    im = im.resize((w,50))
    sdsds = random.uniform(-2,2)
    im = im.rotate(sdsds, Image.BICUBIC)
    img = np.asarray(im)
    img.flags.writeable = True
    img_r = gaussianNoisy(img[:, :, 0].flatten())+random.randint(-5,5)
    img_g = gaussianNoisy(img[:, :, 1].flatten())+random.randint(-5,5)
    img_b = gaussianNoisy(img[:, :, 2].flatten())+random.randint(-5,5)
    s = [0,1,2]
    random.shuffle(s)
    img[:, :, s[0]] = img_r.reshape([50, w])
    img[:, :, s[1]] = img_g.reshape([50, w]) 
    img[:, :, s[2]] = img_b.reshape([50, w])
    img =  Image.fromarray(np.uint8(img))
    im = img
    draw = ImageDraw.Draw(im)
    char_list = list(text.strip())
    start_x, start_y = random.randint(12,30),random.randint(4,6)
    stride = 0
    x_min  = start_x

    font_size = random.randint(19,22)
    font_size_num = random.randint(19,22)
    
    for idx,char in enumerate(text):
        if char in "0123456789":
            #font_path = "/home/robot/hcl/generate_OCRimages/fonts/idcard_address_num.ttf" 
            font_path = "../fonts/idcard_address_num.ttf"
            font = ImageFont.truetype(font_path, font_size)
            x_min = x_min + stride
            stride = int(font_size_num * 0.75)
            if char == "1":
                stride = int(font_size_num * 0.6)
            draw.text(( x_min , start_y+3 ), char,(random.randint(30,60),random.randint(30,60),random.randint(30,60)) ,font=font)
        else:
            font_path = "../fonts/1.ttf"
            #font_path = "/home/robot/hcl/generate_OCRimages/fonts/1.ttf"
            font = ImageFont.truetype(font_path, font_size)
            x_min = x_min + stride
            stride = font_size +random.randint(0,1) 
            draw.text(( x_min , start_y ), char,(random.randint(10,50),random.randint(10,50),random.randint(10,50)) ,font=font)
      

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
    factor = random.randint(4,6)
    new_shape = (cv_im.shape[1] * factor, cv_im.shape[0] * factor)
    im_enlarge = cv2.resize(cv_im,new_shape)
    k = random.choice([3,5])
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
    k = random.choice([[3,1],[5,1],[7,1]])
    result_im = cv2.GaussianBlur(result_im ,( k[0], k[0] ),k[1])
    blur_im = result_im
    
    blur_im = cv2.resize(blur_im, (random.randint(w-2,w+2),random.randint(47,53)))
    im = Image.fromarray(blur_im).rotate(-sdsds, Image.BICUBIC)
    im = im.crop((4,5,260,37))
    cv_im = np.array(im)
    #print(int(36*(len(text)**0.8))-4)`

    cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2GRAY)
    cv_im[:,230:255] = 255
    return cv_im

