import os
import cv2
import numpy as np
import tensorflow as tf

import model
import locality_aware_nms as nms_locality
import lanms
from icdar import restore_rectangle


class TextDetection:
    
    def __init__(self,model_path,GPU,score_map_thresh=0.8,box_thresh=0.1,nms_thresh=0.2,imput_size=(512,512)):
        self.model_path = model_path
        self.score_map_thresh = score_map_thresh
        self.box_thresh = box_thresh
        self.nms_thresh = nms_thresh
        self.model_image_size = imput_size   #(w,h)
        self.gpu = GPU

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        g1 = tf.Graph()
        with g1.as_default():            
            self.input_images = tf.placeholder(tf.float32,shape=[None,None,None,3],name='imput_images')
            #self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.f_score, self.f_geometry = model.model(self.input_images, is_training=False)
            saver = tf.train.Saver()

        self.sess = tf.Session(graph=g1,config=config)
        saver.restore(self.sess,self.model_path)
        # variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())
    
    '''
    def __init__(self,model_path,GPU,score_map_thresh=0.8,box_thresh=0.1,nms_thresh=0.2,imput_size=(512,512)):
        self.model_path = model_path
        self.score_map_thresh = score_map_thresh
        self.box_thresh = box_thresh
        self.nms_thresh = nms_thresh
        self.model_image_size = imput_size   #(w,h)
        self.gpu = GPU

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7           
        self.input_images = tf.placeholder(tf.float32,shape=[None,None,None,3],name='imput_images')
        #self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.f_score, self.f_geometry = model.model(self.input_images, is_training=False)
            
        saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        saver.restore(self.sess,self.model_path)
        # variable_averages = tf.train.ExponentialMovingAverage(0.997, self.global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())    
    '''
    
    def resize_image(self,im):
        h,w,_ = im.shape

        height = self.model_image_size[0]
        width = self.model_image_size[1]
        if w >= h:
            img = cv2.resize(im,(width,int(width*h/w)))
            ratio = width/float(w)
        else:
            img = cv2.resize(im,(int(height*w/h),height))
            ratio = height/float(h)
        # tmp_height = (img.shape[0]//32+1)*32
        # img = cv2.copyMakeBorder(img,0,tmp_height-img.shape[0],0,width-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
        img = cv2.copyMakeBorder(img,0,height-img.shape[0],0,width-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))

        return img,(ratio,ratio)


    def detect(self,score_map, geo_map):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > self.score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
        print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        # nms part
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), self.nms_thresh)
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), self.nms_thresh)

        if boxes.shape[0] == 0:
            return None

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > self.box_thresh]

        return boxes

    def sort_poly(self,p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def get_textboxes(self,im):
        
        #im = cv2.copyMakeBorder(im,0,im.shape[1],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))

        im_resized,(ratio_h,ratio_w) = self.resize_image(im)
        score, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [im_resized[:,:,::-1]]})
        boxes = self.detect(score,geometry)
        bboxes = []
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            for box in boxes:
                box = self.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                bboxes.append(box.astype(np.int32))
        return np.array(bboxes)

    def display_get_textboxes(self,im):
        
        #im = cv2.copyMakeBorder(im,0,im.shape[1],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))

        im_resized,(ratio_h,ratio_w) = self.resize_image(im)
        score, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [im_resized[:,:,::-1]]})
        boxes = self.detect(score,geometry)
        cv2.imshow('im_resized',im_resized)
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            for box in boxes:
                box = self.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
        cv2.imshow('im',im)
        cv2.waitKey(0)

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    textdec = TextDetection('../nice_model/model.ckpt-19176',0)
    im =cv2.imread('../test/test_image/1020160068709378idCardFront.jpg')
    textdec.display_get_textboxes(im)