import base64
import cv2
import json
import numpy as np
import util_code
import text_detection

MODEL_PATH = './nice_model/model.ckpt-19176'

class textDetectMask:
    def __init__(self,gpu):
        try:
            self.textdet = text_detection.TextDetection(MODEL_PATH,gpu)
        except Exception as e:
            print ('textDetectMask init error: ',e)

    def readb64(self,base64_string):
        buf = base64.b64decode(base64_string)
        buf = np.asarray(bytearray(buf), dtype="uint8")
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return image

    def get_polygon_mask(self,image):
        boxes = self.textdet.get_textboxes(image)
        if len(boxes) < 4:
            util_code._mask_area_(image,[0,0,image.shape[1],image.shape[0]])
            print('too few box,so whole image mask!')
        else:
            #img =np.zeros(image.shape,np.uint8)
            img = image.copy()
            util_code._mask_area_(img,[0,0,img.shape[1],img.shape[0]])
            min_y = 100000
            max_y = 0
            max_x1 = 0
            max_x2 = 0
            cv2
            roi = np.zeros(img.shape,dtype=np.uint8)
            for box in boxes:
                p1 = np.min(box,axis=0)
                p2 = np.max(box,axis=0)
                r =  []
                r.extend(p1)
                r.extend(p2-p1)
                cv2.drawContours(roi,[box],0,[255,255,255],-1)
                #img.copy(image,roi)
                if min_y>p2[1]:
                    min_y = p2[1]
                if max_y < p1[1]:
                    max_y = p1[1]
                if p2[0] > max_x1:
                    max_x2 = max_x1
                    max_x1 = p2[0]
                elif p2[0] > max_x2:
                    max_x2 = p2[0]
            w = max_x1-max_x2
            h = max_y-min_y
            off_x = min((image.shape[1]-max_x1)//3,20)
            x1 = max_x2 + w//10
            x2 = max_x1 + off_x
            y1 = min_y + h//20
            y2 = max_y - h//10
            #util_code._mask_area_(image,[x1,y1,x2-x1,y2-y1])
            image[y1:y2,x1:x2] = img[y1:y2,x1:x2]
            roi_inv = cv2.bitwise_not(roi)
            img = cv2.bitwise_and(img,roi)
            image = cv2.bitwise_and(image,roi_inv)
            image = cv2.add(image,img)
            #cv2.rectangle(image,(x1,y1),(x2,y2),[0,0,255],1)
        return image

    def get_mask(self,image):
        boxes = self.textdet.get_textboxes(image)
        if len(boxes) < 4:
            util_code._mask_area_(image,[0,0,image.shape[1],image.shape[0]])
            print('too few box,so whole image mask!')
        else:
            #img =np.zeros(image.shape,np.uint8)
            img = image.copy()
            util_code._mask_area_(img,[0,0,img.shape[1],img.shape[0]])
            min_y = 100000
            max_y = 0
            max_x1 = 0
            max_x2 = 0
            for box in boxes:
                p1 = np.min(box,axis=0)
                p2 = np.max(box,axis=0)
                r =  []
                r.extend(p1)
                r.extend(p2-p1)
                #util_code._mask_area_(image,r)
                image[p1[1]:p2[1],p1[0]:p2[0]]=img[p1[1]:p2[1],p1[0]:p2[0]]
                if min_y>p2[1]:
                    min_y = p2[1]
                if max_y < p1[1]:
                    max_y = p1[1]
                if p2[0] > max_x1:
                    max_x2 = max_x1
                    max_x1 = p2[0]
                elif p2[0] > max_x2:
                    max_x2 = p2[0]
            w = max_x1-max_x2
            h = max_y-min_y
            off_x = min((image.shape[1]-max_x1)//3,20)
            x1 = max_x2 + w//10
            x2 = max_x1 + off_x
            y1 = min_y + h//20
            y2 = max_y - h//10
            #util_code._mask_area_(image,[x1,y1,x2-x1,y2-y1])
            image[y1:y2,x1:x2] = img[y1:y2,x1:x2]

        return image



    def get_result(self,code,data):
        data = json.loads(data)
        app_data = json.loads(data['data'])
        request_id = app_data['request_id']
        image_io = app_data['img']
        try:
            image = self.readb64(image_io)

            if code == '000':
                ret_v = self.get_mask(image)
                bimg = cv2.imencode('.jpg',ret_v)[1]
                base64_data = str(base64.b64encode(bimg))
                return request_id,'000',base64_data
            else:
                return request_id,code,''
        except TypeError as e:
            print ('get_result api TypeError ',e)
            image = None
            return request_id,'002',''

if __name__ == '__main__':
    textmask = textDetectMask(0)
    im = cv2.imread('./test/test_image/3820150003983201idCardFront.jpg')
    cv2.imshow('im',im)
    mask = textmask.get_polygon_mask(im)
    cv2.imshow('mask',mask)
    cv2.waitKey(0)