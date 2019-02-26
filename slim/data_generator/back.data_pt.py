import os
import cv2
import random
import numpy as np

#import keras



class TrainDataGenerator:
    def __init__(self,txtfile, batch_size,pt_num):
        with open(txtfile,'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        self.len = len(lines)
        self.val_num = int(0.1*self.len)
        self.train_num = self.len - self.val_num
        self.train_lines = lines[0:self.train_num]
        self.val_lines = lines[self.train_num:]
        self.batch_size = batch_size
        self.pt_num = pt_num
        self.train_idx = 0
        self.val_idx = 0

    def get_val(self):
        while 1:
            X,Y = self.get_batch('val')
            yield (X,Y)

    def get_train(self):
        while 1:
            X,Y = self.get_batch('train')
            yield (X,Y)

    def get_batch(self,flag):
        X_data = np.zeros((self.batch_size,64,64,3))
        labels = np.zeros((self.batch_size,self.pt_num*2))
        if flag == 'train':
            for i in range(self.batch_size):
                idx = self.train_idx
                X_data[i],labels[i] = self.get_data(idx,flag)
                self.train_idx = self.train_idx + 1
                if self.train_idx == self.train_num:
                    self.train_idx = 0
                    np.random.shuffle(self.train_lines)
        else:
            for i in range(self.batch_size):
                idx = self.val_idx
                X_data[i],labels[i] = self.get_data(idx,flag)
                self.val_idx = self.val_idx + 1
                if self.val_idx == self.val_num:
                    self.val_idx = 0
            
        return X_data,labels

    def get_data(self, i,flag):
        if flag == "train":
            line = self.train_lines[i]
        else:
            line = self.val_lines[i]

        landmarks = []

        image_path = line[0]
        img = cv2.imread(image_path)
        x1 = int(line[1])
        y1 = int(line[2])
        x2 = int(line[3])
        y2 = int(line[4])
        crop_image = img[y1:y2,x1:x2]
        crop_image = cv2.cvtColor(crop_image,cv2.COLOR_BGR2RGB)
        crop_h,crop_w,_ = crop_image.shape
        max_len = max(crop_h,crop_w)

        prob = random.uniform(0,1)
        if prob > 0.5:
            crop_image = np.fliplr(crop_image)
            tmp = np.reshape(line[5:],(-1,2))
            for i in range(4):
                landmarks.append((crop_w - float(tmp[3-i][0]))/max_len)
                landmarks.append((float(tmp[3-i][1]))/max_len)
            landmarks.append((crop_w - float(tmp[4][0]))/max_len)
            landmarks.append((float(tmp[4][1]))/max_len)
            for i in range(2):
                landmarks.append((crop_w - float(tmp[6-i][0]))/max_len)
                landmarks.append((float(tmp[6-i][1]))/max_len)
        else:
           for i in range(self.pt_num*2):
               landmarks.append((float(line[5+i])) /max_len) 
        
        if crop_h >crop_w:
            pad_image = cv2.copyMakeBorder(crop_image,0,0,0,crop_h-crop_w,cv2.BORDER_CONSTANT,value=0)
        else:
            pad_image = cv2.copyMakeBorder(crop_image,0,crop_w-crop_h,0,0,cv2.BORDER_CONSTANT,value=0)
        
        if False:
            for i in range(len(landmarks)/2):
                x = int(landmarks[2*i]*max_len)
                y = int(landmarks[2*i+1]*max_len)
                cv2.circle(pad_image,(x,y),2,(0,0,255),4)
            cv2.imshow("image",pad_image)
            cv2.waitKey(0)

        image = cv2.resize(pad_image,(64,64))
        image = image /255.0
        # for i in range(self.pt_num*2):
        #     landmarks.append((float(line[5+i])) /max_len)
        
        return image,landmarks

if __name__ == "__main__":
    data_gen = TrainDataGenerator('../300VW_pad_anos.txt',32,7)
    data_gen.get_train()