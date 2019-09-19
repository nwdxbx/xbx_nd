import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import model
#from data_generator.data_pro import TrainValDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = "./logs/big_pretrain_gesture/val_loss_3.75806684861_model.ckpt-8760"

def main():
    input_node = tf.placeholder(tf.float32,shape=[None,160,160,3],name="input_images")
    sess = tf.Session()
    # grid_y = np.tile(np.arange(0,10).reshape(-1,1,1,1),[1,10,1,1])
    # grid_x = np.tile(np.arange(0,10).reshape(1,-1,1,1),[10,1,1,1])
    # grid = np.concatenate([grid_x,grid_y],-1)
    f_pred,f_shape = model.model(input_node,is_training=False)
    # import pdb
    # pdb.set_trace()
    tf.contrib.quantize.create_eval_graph()
    saver = tf.train.Saver()
    saver.restore(sess,model_path)

    with open("yolo_eval.pb","w") as f:
        g = tf.get_default_graph()
        f.write(str(g.as_graph_def()))

    saver.save(sess,"./yolo.ckpt-000")


if __name__ == "__main__":
    main()


