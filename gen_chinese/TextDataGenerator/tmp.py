# -*- coding: utf-8 -*-
import cv2
import os,errno
import random
import string
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from computer_text_generator import ComputerTextGenerator
from background_generator import BackgroundGenerator

text = u'中华人民共和国'
font = './1.ttf'
height = 32
text_color = (40,)

# image = ComputerTextGenerator.generate(text, font, text_color, height)



# new_width = int(float(image.size[0] + 10) * (float(height) / float(image.size[1] + 10)))
# resized_img = image.resize((new_width, height - 10), Image.ANTIALIAS)

# background_width = new_width + 10

# background = BackgroundGenerator.gaussian_noise(height, background_width)
# # background = Image.open('test4.jpg')
# # background = background.resize((background_width,height))

# new_text_width, _ = resized_img.size
# background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), 5), resized_img)
# #background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), 5))

# background.convert('RGB').save('./test6.jpg')








background = Image.open('test4.jpg')
image_font = ImageFont.truetype(font,height)
text_width ,text_height = image_font.getsize(text)

background = background.resize((text_width+height,text_height+height))
txt_draw = ImageDraw.Draw(background)
txt_draw.text((height/2,height/2),text,fill=(255,0,0),font=image_font)
rotated_img = background.rotate(5)
background.save('./test.jpg')
rotated_img.save('./test_rotate.jpg')