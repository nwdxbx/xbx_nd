#!/usr/bin/env python
# coding=utf8
import httplib, urllib
import requests
import os
import json
import base64
import httplib
import socket
import glob
import time
import sys
cnt = 0

### Test script for niwodai api

IP="localhost"
def send(img_path,port):
    try:
        url = 'http://localhost:{}/algo/001/'.format(port)

        image = open(img_path, 'rb') #open binary file in read mode
        image_read = image.read()
        image_encode = base64.encodestring(image_read)
        app_data = json.dumps({"request_id": "12345", "img": image_encode})

        print('begin send')
        response = requests.post(url, data={"data": app_data})
        response = json.loads(response.text)
        print('recive finish')
        #print response
        response_data   = response["data"]
        if response_data['code'] == '000':
            with open('test_result/{}.jpg'.format(port),'wb') as f:
                f.write(base64.decodestring(response_data['img']))
            print ('打码成功!')
        else:
            print response_data['message']
            print('打码失败，error!')
    except Exception, e:
        print ("打码失败: ",e)
        return 1  # in case djangle is not accessible

    finally:
        pass


if __name__=='__main__':
    help_msg = 'Usage: python %s <port:8084|8085|8086|8087> ' % sys.argv[0]                      
    if len(sys.argv) != 2:
        print help_msg                                                                            
        sys.exit(1)                                                                               
    port=sys.argv[1]

    os.system('rm -rf test_result/*')
    base_path="test_image"
    img_path=os.path.join(base_path,"111.jpg")
    if not os.path.exists(img_path):
        print 'the filename is not exist.'

    code=send(img_path,port)

