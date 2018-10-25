# -*- coding: utf-8 -*-
from keras.models import Model
import numpy as np
from PIL import Image
import keras.backend  as K
import os
from keras.models import load_model
from keras.layers import Lambda
from custom_charset import charset
from mobilenet_v2 import get_model

def decode(pred):
    batch_size = pred.shape[0] 
    length = pred.shape[1] 
    t = pred.argmax(axis=2)
    char_lists = []
    clsnum = len(charset)
    # import pdb
    # pdb.set_trace()
    for i in range(batch_size):
        char_list = ''
        for ii in range(length):
            prob = t[i]
            if prob[ii] != clsnum and (not (ii > 0 and prob[ii - 1] == prob[ii])):
               char_list = char_list +charset[prob[ii]]
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

bm,m = get_model()
weights_path = "./model_file/deploy.h5"
bm.load_weights(weights_path)
basemodel = bm

def predict(im):
    out = basemodel.predict(im)
    #out = sess.run([basemodel.output],{input:im})[0]
    #print(out.shape)
    y_pred = out[:,2:,:]
    ## print top K 
    '''
    for pred in out:
        labels = []
        for s in pred[:, 1:]:   # pred[:,1:] is thr probability of each char
            c = s.argsort()[-5:][::-1]  #top K
            max11 = np.argmax(s)
            if not(max11 == 5530):
                print(labels_to_text(c+1))
                print(s[c])
            #labels.append(c + 1)
    '''
    '''
    ####
    #---------------only num---------------
    numlabel = [26,93,25,94,632,631,933,29,27,1109,5201,5530,5531]
    a = np.zeros((5532))
    a[numlabel] = 1
    y_pred[-1] = np.multiply(a,y_pred[-1])
    #---------------------------------------
    '''
    #out = K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1],greedy=True)[0][0]
    #out = K.get_value(out)
    out = decode(y_pred)
    return out
