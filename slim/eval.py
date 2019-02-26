import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
input_images = tf.placeholder(tf.float32,shape=[None,64,64,3],name="input_images")
f_preds = model.model(input_images,is_training=False)
img = cv2.imread("test.jpg")
h,w,_=img.shape
max_len=max(h,w)
image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(64,64))
image = image/255.0
image = np.expand_dims(image,0)


sess = tf.Session()
#saver = tf.train.import_meta_graph("./log/111/model.ckpt-330900.meta")
saver = tf.train.Saver()
saver.restore(sess,"./log/666/val_loss_1.10369988201_model.ckpt-347508")
graph = tf.get_default_graph()
results = sess.run(f_preds,feed_dict={input_images:image})
lands = results*max_len
lands=np.reshape(lands,(-1,2))
for idx in range(len(lands)):
    x = int(lands[idx][0])
    y = int(lands[idx][1])
    cv2.circle(img,(x,y),2,(0,255,0),4)
cv2.imshow("img",img)
cv2.imwrite("slim666.jpg",img)
cv2.waitKey(0)
import pdb
pdb.set_trace()
print("finish...")