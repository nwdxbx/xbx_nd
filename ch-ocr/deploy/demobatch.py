#coding:utf-8
#import model
from glob import glob
from PIL import Image
import numpy as np
import time
import preprocessing
import cv2
from ocr.model_test import predict as ocr

paths = glob('./test/hu/*')


if __name__ =='__main__':
    for i in range(1):
        aa = None
        Max = 0
        for path_1 in paths:
            print(path_1)
            im = cv2.imread(path_1)
            h = im.shape[0]
            w = im.shape[1]
            w = int(float(w)/(float(h)/32.0))
            if w > Max:
               Max = w
        #print(Max)
        for path_1 in paths:
            im = cv2.imread(path_1)
            #im = cv2.copyMakeBorder(im,4,5,0,0,cv2.BORDER_CONSTANT,value=(250,250,250))
            h = im.shape[0]
            w = im.shape[1]
            w = int(float(w)/(float(h)/32.0))
            im = cv2.resize(im,(w,32))

            #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            #'''
            im = preprocessing.zmIceColor(im/255.0,3,3)
            im = cv2.cvtColor((im*255.0).astype(np.uint8),cv2.COLOR_BGR2GRAY)
            #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            ##im = preprocessing.zmIceFast(im/255.0,3,3)
            ##im = (im*255.0).astype(np.int)
            #'''
            im = cv2.copyMakeBorder(im,0,0,0,Max-w,cv2.BORDER_CONSTANT,value=255)
            print(im.shape)
            #cv2.imshow("s",im)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            im = im.astype(np.float32)
            im = ((im/255.0)-0.5)*2
            X  = im.reshape((32,Max,1))
            X = np.array([X])
            if aa is None:
               aa = X
               continue
            aa = np.concatenate((aa,X),axis=0)
            #X = np.array([X])
            #print(X.shape)
        #print(aa.shape)
        print("")
        for iiii in range(10):
            t = time.time()
            result = ocr(aa)
            print(time.time()-t)
            print("")
        for ii in result:
            print ii
        for path_1 in paths:
            dd = cv2.imread(path_1)
            cv2.imshow("1",dd)
            cv2.waitKey()
            #cv2.destroyAllWindows()
        print "Time costs:{}s".format(time.time()-t)
        print "---------------------------------------"
