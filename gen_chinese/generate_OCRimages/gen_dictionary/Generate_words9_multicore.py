#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import random
import cv2
import time
import multiprocessing
from words9_gen import gene_img
region_f = codecs.open("Dictionary","r","utf-8")
region_list = region_f.readlines()
length_region_list = len(region_list)
region_f.close()
region_list = [char.strip() for char in region_list]
writer = codecs.open("words9_path","a+","utf-8")
def gen_img(x):
        ss = ""
        for i in range(8):
            ss = ss+random.choice(region_list)
        ss = ss[0:9]
        location = 100
        img = gene_img(ss,location)
        name = str(x)+".jpg"
        img_save_file = os.path.join("/data2/hcl/ocrimgs/dictionary9/", name)
        cv2.imwrite(img_save_file, img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(40,80)])
        writer.write("/data2/hcl/ocrimgs/dictionary9/"+name+","+ss+"\n")
        print(str(x)+" / "+str(2500000))


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = pool.map(gen_img,range(2500000))
    pool.close()
    pool.join()
    #writer.close()
