import os
import cv2
import json
import base64
import numpy as np
from timeit import default_timer as timer

import pynvml
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from utils import cfg
from face_detection import face_detect
from landmark_detection import landmark
from angle_calculation import morphabelAngle

IMG_SIZE_MIN = 48
IMG_SIZE_MAX = 6000
ERROR_MSG_CODE={
    "000":"",
    "002":"IMAGE_ERROR_UNSUPPORTED_FORMAT",
    "003":"IMAGE_SIZE_TOO_SMALL",
    "004":"IMAGE_SIZE_TOO_LARGE",
}

tf_memory_needed=4.0
def GetGPUMemory(gpu):
    """
    specify the gpu index, return total memory
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    TotalMemory = meminfo.total / 1024.0 /  1024.0 / 1024.0
    #print("GPU memory of {}: {}G".format(gpu, TotalMemory))
    return TotalMemory

class H5AntiSpoof:
    def __init__(self,mylog,gpu,port,phase="TEST"):
        try:
            self.log = mylog
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = float(tf_memory_needed) / GetGPUMemory(int(gpu))
            set_session(tf.Session(config=config))

            model = cfg.FACEDETECT_MODEL
            self.face_detection = face_detect.FaceDetection(model,gpu)
            self.log.writelog("INIT CHECK",json.dumps({"flag":"face_detect module init finish"}))

            model = cfg.LANDMARK_MODEL
            self.face_landmark = landmark.faceLandmark(model,gpu)
            self.log.writelog("INIT CHECK",json.dumps({"flag":"face_landmark module init finish"}))

            self.bfm = morphabelAngle.morphabelAngle()
        except Exception as e:
            self.log.writelog("INIT CHECK",json.dumps({"flag":"h5 anti-spoof service init error"}),"ERROR")

    def readb64(self,base64_str):
        buf = base64.b64decode(base64_str)
        buf = np.asarray(bytearray(buf), dtype="uint8")
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return image

    def parse(self,data):
        try:
            the_id = data["request_id"]
        except Exception as e:
            the_id = -1    
        try:
            imageIO1 = data["img1"]
            imageIO2 = data["img2"]
            image1 = self.readb64(imageIO1)
            image2 = self.readb64(imageIO2)
        except Exception as e:
            image1 = None
            image2 = None
            self.log.writelog("IMG PARSE",json.dumps({"flag":"img parse error"}),"ERROR")
        return the_id,image1,image2

    def image_check(self,image1,image2):
        if image1 is None or image2 is None:
            return "002"
        min_size = min(image1.shape[0],image1.shape[1],image2.shape[0],image2.shape[1])
        max_size = max(image1.shape[0],image1.shape[1],image2.shape[0],image2.shape[1])
        if min_size < IMG_SIZE_MIN:
            return "003"
        elif max_size > IMG_SIZE_MAX:
            return "004"
        else:
            return "000"

    def get_face_and_angle(self,the_id,image):
        try:
            start = timer()
            face_dets = self.face_detection.detect([image])
            end = timer()
            self.log.writelog("DETECT TIME",json.dumps({"time":str(end-start)}),request_id=the_id)
        
            if face_dets[0] == []:
                return None,None,"0"
            elif len(face_dets[0]) >= 1:
                y1,x1,y2,x2 = map(int,face_dets[0][0])
                box = [x1,y1,x2,y2]
                start = timer()
                points = self.face_landmark.predict(image,box)
                end = timer()
                self.log.writelog("LANDMARK TIME",json.dumps({"time":str(end-start)}),request_id=the_id)

                start = timer()
                h,w,_ = image.shape
                x = self.bfm.from_image(points,h,w)
                angles = self.bfm.fit(x,max_iter=3)
                end = timer()
                self.log.writelog("FIT TIME",json.dumps({"time":str(end-start)}),request_id=the_id)

                return box,angles,"0"

        except Exception as e:
            self.log.writelog("PROCESS CHECK",json.dumps({"flag":"get_face_and_angle function error!"}),"ERROR",request_id=the_id)
            return None,None,"1"

    def get_status(self,angles1,angles2):
        x1,y1,z1 = angles1
        x2,y2,z2 = angles2
        if abs(y1)<=15 and abs(y2)>=15 and abs(y1-y2)>20:
            return 1
        elif abs(y1)>=15 and abs(y2)<=15 and abs(y1-y2)>20:
            return 1
        else:
            return 0

    def h5_status(self,the_id,image1,image2):
        flag1,angles1,a = self.get_face_and_angle(the_id,image1)
        flag2,angles2,b = self.get_face_and_angle(the_id,image2)
        if flag1==None and flag2==None:
            code = a + b + "11"
            return code,-1,angles1,angles2
        elif flag1==None and flag2!=None:
            code = a + b + "10"
            return code,-1,angles1,angles2
        elif flag1!=None and flag2==None:
            code = a + b + "01"
            return code,-1,angles1,angles2
        else:
            code = "000"
            flag = self.get_status(angles1,angles2)
            return code,flag,angles1,angles2

    def get_verification_result(self,data):
        the_id,image1,image2 = self.parse(data)
        status = self.image_check(image1,image2)
        if status != "000":
            self.log.writelog("IMG CHECK",json.dumps({"flag":ERROR_MSG_CODE[status]}),"WARNING",request_id=the_id)
            result = {"request_id":the_id, "flag":-1, "code":status, "angle1":"", "angle2":"", "message":ERROR_MSG_CODE[status]}
        else:
            message,flag,angles1,angles2 = self.h5_status(the_id,image1,image2)
            result = {"request_id":the_id, "flag":flag, "code":status, "angle1":str(angles1), "angle2":str(angles2), "message":message}
        self.log.writelog("PROCESS RESULT",json.dumps({"RESULT":str(result)}),request_id=the_id)
        return result


