import os
import cv2
import numpy as np
from face_detection import face_detect
from landmark_detection import landmark
from angle_calculation import morphabelAngle

def main():
    face_model = "/opt/models/h5_anti_spoof/model/yolo3.h5"
    land_model = "/opt/models/h5_anti_spoof/model/256_256_resfcn256_weight"
    #bfm_model = "/opt/models/h5_spoof/model/BFM.mat"
    Det = face_detect.FaceDetection(face_model,0)
    LandDet = landmark.faceLandmark(land_model,0)
    bfm = morphabelAngle.morphabelAngle()
    path = "./0"
    files = os.listdir(path)
    for name in files:
        filename = os.path.join(path,name)
        image = cv2.imread(filename)
        h,w,_ = image.shape
        boxes = Det.detect([image])
        if len(boxes[0])!=0:
            y1,x1,y2,x2 = map(int,boxes[0][0])
            box = [x1,y1,x2,y2]
            landmarks = LandDet.predict(image,box)
            x = bfm.from_image(landmarks,h,w)
            angles = bfm.fit(x,max_iter=3)
            str_x = "thea_x: %.2f" % (angles[0])
            str_y = "thea_y: %.2f" % (angles[1])
            str_z = "thea_z: %.2f" % (angles[2])
            cv2.putText(image,str_x,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
            cv2.putText(image,str_y,(10,80),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
            cv2.putText(image,str_z,(10,120),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.imshow("image",image)
        cv2.waitKey(0)

def list_cap():
    face_model = "/opt/models/h5_anti_spoof/model/yolo3.h5"
    land_model = "/opt/models/h5_anti_spoof/model/256_256_resfcn256_weight"
    Det = face_detect.FaceDetection(face_model,0)
    LandDet = landmark.faceLandmark(land_model,0)
    bfm = morphabelAngle.morphabelAngle()
    srcdirs = "/data_c/datasets/train_8000_3/Foreign/SDK/user_timeout1"
    lines = os.listdir(srcdirs)
    idx = [30,36,39,42,45,48,54]
    for line in lines:
        video = os.path.join(srcdirs,line)
        cap = cv2.VideoCapture(video)
        while True:
            success,image = cap.read()
            if not success:
                break
            if image is None:
                continue
            image = np.rot90(image)
            image = np.ascontiguousarray(image, dtype=np.uint8)
            h,w,_ = image.shape
            boxes = Det.detect([image])
            if len(boxes[0])!=0:
                y1,x1,y2,x2 = map(int,boxes[0][0])
                box = [x1,y1,x2,y2]
                landmarks = LandDet.predict(image,box)
                for i in range(len(landmarks)):
                    if i in idx:
                        point = landmarks[i]
                        cx,cy = map(int,point)
                        cv2.circle(image,(cx,cy),1,(255,255,0),2)
                x = bfm.from_image(landmarks,h,w)
                angles = bfm.fit(x,max_iter=3)
                str_x = "thea_x: %.2f" % (angles[0])
                str_y = "thea_y: %.2f" % (angles[1])
                str_z = "thea_z: %.2f" % (angles[2])
                cv2.putText(image,str_x,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
                cv2.putText(image,str_y,(10,80),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
                cv2.putText(image,str_z,(10,120),cv2.FONT_HERSHEY_SIMPLEX,1.,(0,0,255),2)
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.imshow("image",image)
            cv2.waitKey(0)


if __name__ == "__main__":
    list_cap()