## keras == 2.1.5
from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D,ZeroPadding2D
from keras.layers import Input, Conv2D,Conv1D
from keras.layers import Flatten,Permute,TimeDistributed,Bidirectional,GRU,Dropout,LSTM
from keras.layers import Activation, Dense,add
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import time
from sru import SRU

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def InvertedResidualBlock(x, expand, out_channels, repeats, stride, block_id,kernel_size = (3,3),padding = 'same'):

    channel_axis = -1
    in_channels = K.int_shape(x)[-1]
    x = Conv2D(expand * in_channels, 1, padding=padding, strides=stride, use_bias=False, activation='relu',name='conv_%d_0' % block_id)(x)
    x = BatchNormalization(axis=-1,name='conv_%d_0_bn' % block_id)(x)
    x = DepthwiseConv2D(kernel_size,padding='same',depth_multiplier=1,strides=1,use_bias=False,name='conv_dw_%d_0' % block_id )(x)
    x = BatchNormalization(axis=-1,name='conv_dw_%d_0_bn' % block_id)(x)
    x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False, activation='relu',name='conv_bottleneck_%d_0' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_bottlenet_%d_0_bn' % block_id)(x)

    for i in xrange(1, repeats):
        x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,activation='relu',name='conv_%d_%d' % (block_id, i))(x)
        x1 = BatchNormalization(axis=-1,name='conv_%d_%d_bn' % (block_id, i))(x1)
        x1 = DepthwiseConv2D(kernel_size,padding='same',depth_multiplier=1,strides=1,use_bias=False,name='conv_dw_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=-1,name='conv_dw_%d_%d_bn' % (block_id, i))(x1)
        x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,activation='relu',name='conv_bottleneck_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=-1,name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)
        x = add([x, x1], name='block_%d_%d_output' % (block_id, i))
    return x


def get_model(height=32,nclass=5532):

    rnnunit = 256
    #img_input = Input(shape=(height,None,1),name='the_input')
    img_input = Input(shape=(height,None,1),name='the_input')
    #img_input = Input(shape=(1,height,256),name='the_input')
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),activation='relu', name='conv1')(img_input)
    x = ZeroPadding2D(padding=(0,1))(x)
    x = InvertedResidualBlock(x, expand=6, out_channels=64, repeats=1, stride=(2,2), block_id=1)
    x = InvertedResidualBlock(x, expand=6, out_channels=128, repeats=2, stride=(2,1), block_id=2)
    x = InvertedResidualBlock(x, expand=6, out_channels=256, repeats=2, stride=(2,1), block_id=3)
    x = InvertedResidualBlock(x, expand=6, out_channels=512, repeats=2, stride=(2,1), block_id=4,kernel_size = (2,2))
    x = Permute((2,1,3),name='permute1')(x)
    #x = Permute((3,2,1),name='permute1')(x)
    x = TimeDistributed(Flatten(),name='timedistrib')(x)
    #x = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(x)
    #x = GRU(rnnunit,return_sequences=True)(x)
    x = SRU(rnnunit,return_sequences=True)(x)
    #x = Bidirectional(SRU(rnnunit,return_sequences=True),name='blstm1')(x)
    '''
    #x = Permute((2,1),name='permute2')(x)
    #my_expand = Lambda(lambda x: K.expand_dims(x, axis=-1))
    #x = my_expand(x)
    y_pred = Conv1D(nclass,kernel_size=1,padding='same',strides=1,activation='softmax')(x)
    #my_squeeze = Lambda(lambda x: K.squeeze(x, axis=-1))
    #y_pred = my_squeeze(x)
    '''
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(x)
    basemodel = Model(inputs=img_input,outputs=y_pred)

    labels = Input(name='the_labels', shape=[None,], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[img_input, labels, input_length, label_length], outputs=[loss_out])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.summary()

    return basemodel,model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model,basemodel = get_model(32,5531)
    for i in range(10):
        ii = np.random.randn(1,32,256,1)
        #ii = np.random.randn(1,1,32,256)
        t = time.time()
        ss = basemodel.predict(ii)
        print(time.time()-t)
    ddd = 0
    for i in range(30):
        ii = np.random.randn(1,32,256,1)
        #ii = np.random.randn(1,1,32,256)
        t = time.time()
        ss = basemodel.predict(ii)
        ddd = time.time()-t + ddd
    print(ddd/30.0)
