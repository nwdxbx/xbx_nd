#! /usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import codecs
import random
import cv2
import time
from train_gen_alpha import gene_img
import motionBlur

if __name__ == "__main__":
    alphaf = "ABCDEFGHXYZ"
    region_f = codecs.open("city_remove","r","utf-8")
    region_list = region_f.readlines()
    length_region_list = len(region_list)
    region_f.close()
    region_list = [dd.strip() for dd in region_list]
    Path_String_Writer = codecs.open("city_alpha_blur", "a+", "utf-8")
    for i in range(100000):
            line = random.choice(region_list)+random.choice(region_list)+random.choice(region_list)+random.choice(region_list)+random.choice(region_list)
            line = line[0:8]
            line = line + random.choice(alphaf)
            img = gene_img(line) 
            kernel,anchor=motionBlur.genaratePsf(random.randint(4,7),random.randint(-179,179))
            try:
                img=cv2.filter2D(img,-1,kernel,anchor=anchor)
            except:
                print "error"
            name = str(i)+".jpg"
            #cv2.imshow("img", img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Path_String_Writer.write("/data2/hcl/ocrimgs/city_alpha_blur/"+ name +","+line+"\n")
            img_save_file = os.path.join("/data2/hcl/ocrimgs/city_alpha_blur", name)
            cv2.imwrite(img_save_file, img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(20,50)])
            print(str(i)+" / "+str(100000))
Path_String_Writer.close()
