#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import random
import cv2
import time
from train_img_gene_soft_city import gene_img


if __name__ == "__main__":
    alphaf = ["0","1","2","3","4","5","6","7","8","9","0"]
    region_f = codecs.open("city","r","utf-8")
    region_list = region_f.readlines()
    length_region_list = len(region_list)
    region_f.close()
    region_list = [dd.strip() for dd in region_list]
    Path_String_Writer = codecs.open("city_num_hyphen_bracket", "a+", "utf-8")
    for i in range(100000,150000):
            line = random.choice(region_list)+random.choice(region_list)+random.choice(region_list)
            line = line[0:2]
            line = line + random.choice(alphaf) +"("+ random.choice(alphaf)+")" +random.choice(alphaf) + "-"+ random.choice(alphaf)
            img = gene_img(line) 
            name = str(i)+".jpg"
            #cv2.imshow("img", img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Path_String_Writer.write("/data2/hcl/ocrimgs/city_num_hyphen_bracket/"+ name +","+line+"\n")
            img_save_file = os.path.join("/data2/hcl/ocrimgs/city_num_hyphen_bracket/", name)
            cv2.imwrite(img_save_file, img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(25,60)])
            print(str(i)+" / "+str(150000))
    Path_String_Writer.close()
