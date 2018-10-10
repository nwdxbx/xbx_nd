#! /usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import codecs
import random
import cv2
import time
from train_img_road import gene_img


if __name__ == "__main__":
    number = "0123456789"
    pick = [u"村",u"弄",u"巷",u"号",u"栋",u"室",u"单元",u"房",u"屯",u"组"]
    region_f = codecs.open("Road_Name.txt","r","utf-8")
    region_list = region_f.readlines()
    length_region_list = len(region_list)
    region_f.close()
    region_list = [dd.strip() for dd in region_list]
    Path_String_Writer = codecs.open("road", "a+", "utf-8")
    for i in range(200000):
            line = random.choice(region_list)
            for iii in range(5):
                line = line + str(random.randint(1,99))+random.choice(pick)
            line = line[0:9]
            img = gene_img(line) 
            name = str(i)+".jpg"
            #cv2.imshow("img", img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Path_String_Writer.write("/data2/hcl/ocrimgs/road/"+ name +","+line+"\n")
            img_save_file = os.path.join("/data2/hcl/ocrimgs/road/", name)
            cv2.imwrite(img_save_file, img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(20,60)])
            print(str(i)+" / "+str(2000000))	
Path_String_Writer.close()
