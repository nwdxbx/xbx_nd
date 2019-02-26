import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import model
from data_generator.data_pt import TrainDataGenerator

tf.app.flags.DEFINE_integer("kp", 7, "")
tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_string("annotation_path", "./300VW_anos.txt", "")

FLAGS = tf.app.flags.FLAGS

input_images = tf.placeholder(tf.float32,shape=[None,64,64,3],name="input_images")
sess = tf.Session()
f_preds = model.model(input_images,is_training=False)
#data_gen = TrainDataGenerator(FLAGS.annotation_path,FLAGS.batch_size,FLAGS.kp)

tf.contrib.quantize.create_eval_graph()

saver = tf.train.Saver()
saver.restore(sess,"./log/666/val_loss_1.10369988201_model.ckpt-347508")

# for step in range(700):
#     val_generator = data_gen.get_val()
#     val_data = next(val_generator)
#     resuluts = sess.run(f_preds,feed_dict={input_images:val_data[0]})
#     print("results: ",resuluts)
with open("landmark_eval.pb","w") as f:
    g = tf.get_default_graph()
    # import pdb
    # pdb.set_trace()
    f.write(str(g.as_graph_def()))

saver.save(sess,"./model.ckpt-358540")
