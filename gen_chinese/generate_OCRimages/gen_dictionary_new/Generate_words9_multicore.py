#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import random
import cv2
import time
import multiprocessing
from words9_gen import gene_img
import motionBlur
#region_f = codecs.open("Dictionary","r","utf-8")
region_f = codecs.open("Dictionary_new","r","utf-8")
region_list = region_f.readlines()
length_region_list = len(region_list)
region_f.close()
region_list = [char.strip() for char in region_list]
writer = codecs.open("words9_path","a+","utf-8")
notin_reader = codecs.open("notin","r","utf-8")
notin = notin_reader.readlines()
notin_reader.close()
o = [dd.strip() for dd in notin]
def gen_img(x):
        ss = ""
        for i in range(8):
            ss = ss+random.choice(region_list)
        for dd in o:
            ss = ss.replace(dd,"")
        ss = ss[0:9]
        img = gene_img(ss)
        kernel,anchor=motionBlur.genaratePsf(random.randint(4,7),random.randint(-179,179))
        try:
            img=cv2.filter2D(img,-1,kernel,anchor=anchor)
        except:
            print "error"
        name = str(x)+".jpg"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        img_save_file = os.path.join("/data2/hcl/ocrimgs/newdictionary9/", name)
        cv2.imwrite(img_save_file, img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(30,60)])
        writer.write("/data2/hcl/ocrimgs/newdictionary9/"+name+","+ss+"\n")
        print(str(x)+" / "+str(750000))


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = pool.map(gen_img,range(750000))
    pool.close()
    pool.join()
    #writer.close()
