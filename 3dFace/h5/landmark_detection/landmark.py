import cv2
import numpy as np
from predictor import PosPrediction


class faceLandmark(object):
    def __init__(self,model,gpu,resolution = 256):
        self.resa = self.resb = 256
        self.pos_preditor = PosPrediction(gpu,self.resa,self.resb)
        self.pos_preditor.restore(model)
        self.faceindex = np.array([24591, 30230, 36122, 42272, 46893, 48707, 48219, 47984, 49536,
                                    48015, 48292, 48828, 47058, 42463, 36325, 30441, 24816, 12602,
                                    10823, 10069, 10337, 10858, 10901, 10398, 10154, 10936, 12741,
                                    15232, 18816, 22144, 24704, 28533, 29050, 29568, 29061, 28554,
                                    17230, 15446, 15711, 16742, 17504, 17751, 16793, 15776, 15529,
                                    17329, 17832, 17567, 36460, 33652, 32636, 32896, 32643, 33675,
                                    36498, 38025, 38532, 38528, 38523, 38006, 36206, 34682, 34432,
                                    34693, 36497, 36740, 36480, 36731],dtype=np.int32)

    def circlebox(self,img,box):
        x1,y1,x2,y2 = box
        oldsize = (x2+y2-x1-y1)/2
        center = np.array([(x1+x2)/2, (y1+y2)/2+int(0.1*oldsize)])
        size = int(oldsize*1.5)
        SRC_PTS = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,256 - 1], [256 - 1, 0]])
        tform = cv2.estimateRigidTransform(SRC_PTS,DST_PTS,False)
        image = cv2.warpAffine(img,tform,(self.resa,self.resa))

        return image,tform

    def predict(self,img,box):
        h, w, _ = img.shape
        image ,tform = self.circlebox(img,box)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image/255.
        pos = np.reshape(self.pos_preditor.predict(image),(-1,3)).T
        pos[2,:] = 1
        tform = np.vstack((tform,np.array([0,0,1])))
        pos = np.dot(np.linalg.inv(tform),pos).T.reshape(-1,3)
        landmarks = pos[self.faceindex,:2].astype(np.float32)

        return landmarks



