#!/usr/bin/env python  
#encoding: utf-8  
#Author: xiao hui
#Date:2017.04.01
#function: functions that used to get the code position 
#tips:
#1. 检测的头像下沿线应该在检测到的号码的上方，而不能有交集。
#2. 检测的头像下沿线的方向和检测到的号码的方向差不多一致。
#3. 若号码检测不到,则以头像下沿线方向为准。
#4. 小波掩码。
import os
import cv2  
import copy 
import math 
import time 
import numpy as np  
import glob

import pywt 
import  cv2

import re


def check_code(data):
    max_len=12
    gap=[10,18]
    data=np.abs(data)
    _iter=0
    ret=[]
    for i in range(0,len(data)-max_len):
        if np.sum(data[i:i+max_len])!=0:
            if _iter>0:
                ret.append([i,_iter])
                _iter=0
        else:
            if len(ret)>0 and _iter==0:
                ret.append([i,_iter])
            else:
                _iter+=1
                continue
    if len(ret)==1:
        return ret[0][0]+max_len,ret[0][0]+30,True
    elif len(ret)>1:
        return ret[0][0]+max_len,ret[1][0],True if ret[1][0]-(ret[0][0]+max_len)>=gap[0] and ret[1][0]-(ret[0][0]+max_len)<gap[1] else False
    else:
        return None,None,False

def _rotate2d_(degrees,point,origin):
    x = point[0] - origin[0]
    yorz = point[1] - origin[1]
    newx = (x*np.cos(math.radians(degrees))) - (yorz*np.sin(math.radians(degrees)))
    newyorz = (x*np.sin(math.radians(degrees))) + (yorz*np.cos(math.radians(degrees)))
    newx += origin[0]
    newyorz += origin[1] 

    return int(newx),int(newyorz)

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def _code_parser_(img0):
    h,w=img0.shape[0],img0.shape[1]

    img1 = copy.deepcopy(img0)
    r=None
    degs=[0,5,-5,10,-10,15,-15,20,-20,25,-25,30,-30,35,-35,40,-40]
    for deg in degs:
        img = copy.deepcopy(img0)
        imgg=copy.deepcopy(img)


        img = cv2.blur(img, (5,5))
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        img=255-img

        if deg!=0:
            img=rotate_about_center(img, deg, scale=1.)
            img=img[int(0.205*img.shape[0]):int((1-0.205)*img.shape[0]),int(0.205*img.shape[1]):int((1-0.205)*img.shape[1])]

        ret,img=cv2.threshold(img,180,255,cv2.THRESH_BINARY)
        pywt_level=5
        coeffs= pywt.wavedec2(img, 'haar', level=pywt_level)

        _coeffs=copy.deepcopy(coeffs)
        pp=np.sum(coeffs[pywt_level][1],1)
        a1,a2,r=check_code(pp)
        if r:
            y1=int(h*(float(a1)/len(pp)))
            y2=int(h*(float(a2)/len(pp)))

            x11,y11=_rotate2d_(-deg,[0, y1],[w/2,h/2])
            x22,y22=_rotate2d_(-deg,[w, y2],[w/2,h/2])

            '''
            img3=img0[y11:y22,0:w]
            NpKernel = np.uint8(np.ones((20,20)))  
            img3 = cv2.erode(img3,NpKernel)  
            print("***************%s"%(deg))
            '''
            #print(y22,y11)
            return [0, y11-10,w, np.abs(y22-y11)+20]
    return [None,None,None,None]

def _mask_area_gray_(masked_img,rect,scale=True):

    _mi=masked_img[:,:,0][int(rect[1]):int(rect[1]+rect[3]),int(rect[0]):int(rect[0]+rect[2])]

    if scale:
        mi=cv2.resize(_mi,(64,64*_mi.shape[0]/_mi.shape[1]),interpolation=cv2.INTER_LINEAR)
    else:
        mi=_mi
    coeffs= pywt.wavedec2(mi, 'haar', level=2)
    for i in range(0,2):
        coeffs[i+1][0][:,:]=0
        coeffs[i+1][1][:,:]=0
        coeffs[i+1][2][:,:]=0
    img=pywt.waverec2(coeffs, 'haar')

    if scale:
        img=cv2.resize(img,(_mi.shape[1],_mi.shape[0]),interpolation=cv2.INTER_LINEAR)

    y1=int(rect[1])
    y2=rect[1]+img.shape[0]
    x1=int(rect[0])
    x2=rect[0]+img.shape[1]
    y2=int(y2 if y2<masked_img.shape[0] else masked_img.shape[0])
    x2=int(x2 if x2<masked_img.shape[1] else masked_img.shape[1])
    masked_img[y1:y2,x1:x2,0]=img[0:y2-y1,0:x2-x1]
    masked_img[y1:y2,x1:x2,1]=img[0:y2-y1,0:x2-x1]
    masked_img[y1:y2,x1:x2,2]=img[0:y2-y1,0:x2-x1]

def _mask_area_(masked_img,rect,scale=True,scale_size=64):
     #b,g,r = cv2.split(masked_img)
    if rect[1]>masked_img.shape[0]:
        return
    if rect[0]>masked_img.shape[1]:
        return
    if rect[3]+rect[1]>masked_img.shape[0]:
        rect[3]=masked_img.shape[0]-rect[3]
    if rect[2]+rect[0]>masked_img.shape[1]:
        rect[2]=masked_img.shape[0]-rect[2]

    rect[0]=0 if rect[0]<0 else rect[0]
    rect[1]=0 if rect[1]<0 else rect[1]

    rect[2]=10 if rect[2]<10 else rect[2]
    rect[3]=10 if rect[3]<10 else rect[3]

    _b_mi = masked_img[:,:,0][int(rect[1]):int(rect[1]+rect[3]),int(rect[0]):int(rect[0]+rect[2])]
    _g_mi = masked_img[:,:,1][int(rect[1]):int(rect[1]+rect[3]),int(rect[0]):int(rect[0]+rect[2])]
    _r_mi = masked_img[:,:,2][int(rect[1]):int(rect[1]+rect[3]),int(rect[0]):int(rect[0]+rect[2])]

    if scale:
        b_mi=cv2.resize(_b_mi,(scale_size,scale_size*_b_mi.shape[0]/_b_mi.shape[1]),interpolation=cv2.INTER_LINEAR)
        g_mi=cv2.resize(_g_mi,(scale_size,scale_size*_g_mi.shape[0]/_g_mi.shape[1]),interpolation=cv2.INTER_LINEAR)
        r_mi=cv2.resize(_r_mi,(scale_size,scale_size*_r_mi.shape[0]/_r_mi.shape[1]),interpolation=cv2.INTER_LINEAR)
    else:
        b_mi=_b_mi
        g_mi=_g_mi
        r_mi=_r_mi

    coeffs_b= pywt.wavedec2(b_mi, 'haar', level=2)
    for i in range(0,2):
        coeffs_b[i+1][0][:,:]=0
        coeffs_b[i+1][1][:,:]=0
        coeffs_b[i+1][2][:,:]=0
    img_b=pywt.waverec2(coeffs_b, 'haar')

    if scale:
        img_b=cv2.resize(img_b,(_b_mi.shape[1],_b_mi.shape[0]),interpolation=cv2.INTER_LINEAR)

    coeffs_g= pywt.wavedec2(g_mi, 'haar', level=2)
    for i in range(0,2):
        coeffs_g[i+1][0][:,:]=0
        coeffs_g[i+1][1][:,:]=0
        coeffs_g[i+1][2][:,:]=0
    img_g=pywt.waverec2(coeffs_g, 'haar')

    if scale:
        img_g=cv2.resize(img_g,(_g_mi.shape[1],_g_mi.shape[0]),interpolation=cv2.INTER_LINEAR)

    coeffs_r= pywt.wavedec2(r_mi, 'haar', level=2)
    for i in range(0,2):
        coeffs_r[i+1][0][:,:]=0
        coeffs_r[i+1][1][:,:]=0
        coeffs_r[i+1][2][:,:]=0
    img_r=pywt.waverec2(coeffs_r, 'haar')

    if scale:
        img_r=cv2.resize(img_r,(_r_mi.shape[1],_r_mi.shape[0]),interpolation=cv2.INTER_LINEAR)
    
    y1=int(rect[1])
    y2=rect[1]+img_b.shape[0]
    x1=int(rect[0])
    x2=rect[0]+img_b.shape[1]
    y2=int(y2 if y2<masked_img.shape[0] else masked_img.shape[0])
    x2=int(x2 if x2<masked_img.shape[1] else masked_img.shape[1])
    masked_img[y1:y2,x1:x2,0]=img_b[0:y2-y1,0:x2-x1]
    masked_img[y1:y2,x1:x2,1]=img_g[0:y2-y1,0:x2-x1]
    masked_img[y1:y2,x1:x2,2]=img_r[0:y2-y1,0:x2-x1]

