# -*- coding: utf-8 -*-
from keras.models import Model
import numpy as np
from PIL import Image
import keras.backend  as K
import os
from keras.models import load_model
from keras.layers import Lambda
charset = "0123456789:"
def decode(pred):
    batch_size = pred.shape[0] 
    length = pred.shape[1] 
    t = pred.argmax(axis=2)
    char_lists = []
    n = len(charset)
    for i in range(batch_size):
        char_list = ''
        for ii in range(length):
            c = t[i]
            if c[ii] != n and (not (ii > 0 and c[ii - 1] == c[ii])):
               char_list = char_list +charset[c[ii]]
        char_lists.append(char_list) 
    return char_lists

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
modelPath = "/data/ocrmodel/keras_yinni_num_maohao_model/basemodel05.h5"
basemodel = load_model(modelPath)
def predict(im):
    out = basemodel.predict(im)
    y_pred = out[:,2:,:]
    out = decode(y_pred)
    return out
