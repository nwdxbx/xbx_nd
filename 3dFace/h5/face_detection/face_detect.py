# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import os
import cv2
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.backend.tensorflow_backend import set_session,get_session

from model import yolo_eval
"""
from model import yolo_body, tiny_yolo_body
"""
from utils import letterbox_image
import json

class FaceDetection(object):
    def __init__(self,model,gpu):
        self.iou = 0.3
        self.score = 0.5
        self.model_path = model
        self.class_names = ["face"]
        self.batchsize = 1
        self.model_image_size = (160, 160) # fixed size or (None, None), hw
        self.anchors = np.array([22.,43.,85.,84.,105.,133.]).reshape(-1,2)

#        config = tf.ConfigProto()
#        config.gpu_options.per_process_gpu_memory_fraction = 0.1
#        self.sess = tf.Session(config=config)
#        set_session(self.sess) 
        self.sess = get_session()
        #self.generate()      
        self.boxes, self.scores, self.classes = self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting

        self.yolo_model = load_model(model_path, compile=False)
        assert self.yolo_model.layers[-1].output_shape[-1] == \
            num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'
        """
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        """

        #print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(None,2))
        boxes, scores, classes = yolo_eval([self.yolo_model.output], self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou,batchsize = self.batchsize)
        return boxes, scores, classes

    def detect(self,images):
        out_boxes = []
        out_scores = []
        out_classes = []
        for image in images:
            image_datas = []
            image_shapes = []
            if self.model_image_size != (None, None):
                assert self.model_image_size[0]%16 == 0, 'Multiples of 32 required'
                assert self.model_image_size[1]%16 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(image,self.model_image_size)
            """
            else:
                new_image_size = (image.width - (image.width % 16),
                                image.height - (image.height % 16))
                boxed_image = letterbox_image(image,self.model_image_size)
            """
            boxed_image = cv2.cvtColor(boxed_image,cv2.COLOR_BGR2RGB)
            image_data = np.array(boxed_image, dtype='float32')

            image_data /= 255.
            image_datas.append(image_data)
            image_shapes.append([image.shape[0], image.shape[1]])

            start = timer()
            boxes, scores, classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_datas,
                    self.input_image_shape: image_shapes,
                    K.learning_phase(): 0
                })
            out_boxes.append(boxes[0].tolist())
            out_scores.append(scores[0].tolist())
            out_classes.append(classes[0].tolist())

        end = timer()
        #print(end - start)

        return out_boxes

    """
    def batch_detect(self,images):
        image_datas = []
        image_shapes = []
        # batchsize = len(images)
        # self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
        #         len(self.class_names), self.input_image_shape,batchsize,
        #         score_threshold=self.score, iou_threshold=self.iou)
        for image in images:
            if self.model_image_size != (None, None):
                assert self.model_image_size[0]%16 == 0, 'Multiples of 32 required'
                assert self.model_image_size[1]%16 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(image,self.model_image_size)
            else:
                new_image_size = (image.width - (image.width % 16),
                                image.height - (image.height % 16))
                boxed_image = letterbox_image(image,self.model_image_size)
            boxed_image = cv2.cvtColor(boxed_image,cv2.COLOR_BGR2RGB)
            image_data = np.array(boxed_image, dtype='float32')

            image_data /= 255.
            #image_data = np.expand_dims(image_data, 0)
            image_datas.append(image_data)
            image_shapes.append([image.shape[0], image.shape[1]])

            start = timer()

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_datas,
                self.input_image_shape: image_shapes,
                K.learning_phase(): 0
            })
        end = timer()
        #print(end - start)

        return out_boxes

    def close_session(self):
        self.sess.close()
    """