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

im = Image.open("background_1.jpg")
print(im.size)
for i in range(5):
	w = random.randint(0,im.size[0]-25)
	h = random.randint(0,im.size[1]-25)
	print(w,h)
	img = im.crop((w,h,w+25,h+25))
	img.show()
