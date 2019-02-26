import tensorflow as tf
from tensorflow.contrib import slim

def inverted_bottleneck(x,expand,out_channels,repeats,stride,block_id):
    in_channels = x.get_shape().as_list()[-1]
    x = slim.conv2d(x,expand*in_channels,1,stride=stride,scope="conv_%d_0" % block_id)
    #x = slim.batch_norm(x,decay=0.99,scale=True,epsilon=0.001,is_training=True)
    x = slim.separable_conv2d(x,None,3,1,scope="conv_dw_%d_0" % block_id)
    #x = slim.batch_norm(x,decay=0.99,scale=True,epsilon=0.001,is_training=True)
    x = slim.conv2d(x,out_channels,1,activation_fn=None,scope="conv_bottleneck_%d_0" % block_id)
    #x = slim.batch_norm(x,decay=0.99,scale=True,epsilon=0.001,is_training=True)

    for i in xrange(1,repeats):
        x1 = slim.conv2d(x,expand*out_channels,1,scope="conv_%d_%d" % (block_id,i))
        #x1 = slim.batch_norm(x1,decay=0.99,scale=True,epsilon=0.001,is_training=True)
        x1 = slim.separable_conv2d(x1,None,3,1,scope="conv_dw_%d_%d" % (block_id,i))
        #x1 = slim.batch_norm(x1,decay=0.99,scale=True,epsilon=0.001,is_training=True)
        x1 = slim.conv2d(x1,out_channels,1,activation_fn=None,scope="conv_bottleneck_%d_%d" % (block_id,i))
        #x1 = slim.batch_norm(x1,decay=0.99,scale=True,epsilon=0.001,is_training=True)
        #x = tf.add(x,x1,name="block_%d_%d_output" % (block_id,i))
        x = x + x1
        x = tf.identity(x,name="block_%d_%d_output" % (block_id,i))

    return x


def model(images,kp=7,is_training=True):
    batch_norm_params = {
        'decay': 0.99,
        'epsilon': 0.001,
        'scale': True,
        'is_training': is_training
        }
    with slim.arg_scope([slim.conv2d,slim.separable_conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn= slim.batch_norm,
                        normalizer_params=batch_norm_params):
        #biases_initializer=None, ,activation_fn=tf.nn.relu,padding="SAME",normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer=slim.l2_regularizer(0.00005)
        x = slim.conv2d(images,16,[3,3],stride=2,scope="conv1")
        x = inverted_bottleneck(x,expand=6,out_channels=16,repeats=2,stride=2,block_id=1)
        x = inverted_bottleneck(x,expand=6,out_channels=32,repeats=2,stride=2,block_id=2)
        x = inverted_bottleneck(x,expand=6,out_channels=64,repeats=2,stride=2,block_id=3)
        x = slim.conv2d(x,256,1,normalizer_fn=None,normalizer_params=None,scope="1x1_out")
        
        with tf.variable_scope("logits"):
            kernel1 = x.get_shape().as_list()[1]
            kernel2 = x.get_shape().as_list()[2]
            x= slim.avg_pool2d(x,[kernel1,kernel2])
            # x = slim.conv2d(x,kp*2,1,normalizer_fn=None,normalizer_params=None,activation_fn=None,scope="conv2d_1c_1x1")
            # y = tf.squeeze(x,[1,2],name="out")
            


            x = slim.flatten(x,scope="flatten")
            y = slim.fully_connected(x,kp*2,activation_fn=None,scope="pred")
            y = tf.identity(y,name="out")
            # _ = slim.softmax(y,scope="predictions")
        
            #prediction_dict = {'logits':y}
            return y