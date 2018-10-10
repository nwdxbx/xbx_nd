#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import random
import cv2
import time
from num_gen import gene_img
import copy
import motionBlur

alph = "0123456789X"

if __name__ == "__main__":
       Path_String_Writer = codecs.open("num9_blur", "a+")
       start,end = 0,100000
       for i in range(start,end):
               idnum = ""
               for ii in range(9):
                   idnum = idnum+random.choice(alph)
               img = gene_img(idnum)
               kernel,anchor=motionBlur.genaratePsf(random.randint(4,7),random.randint(-179,179))
               try:
                   img=cv2.filter2D(img,-1,kernel,anchor=anchor)
               except:
                   print("error")

               name = str(i)+".jpg"

               #cv2.imshow("img", img)
               #cv2.waitKey()
               #cv2.destroyAllWindows()
               #print(idnum)
               Path_String_Writer.write("/data2/hcl/ocrimgs/num9_blur/"+name+","+idnum+"\n")
	       img_save_file = os.path.join("/data2/hcl/ocrimgs/num9_blur/", name)
	       cv2.imwrite(img_save_file, img,[int(cv2.IMWRITE_JPEG_QUALITY),random.randint(25,50)])
               print(str(i+1)+" /"+ str(end-start))	
       Path_String_Writer.close()
