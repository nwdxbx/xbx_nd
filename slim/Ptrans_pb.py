import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
import model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_path = './log/000/model.ckpt-3600'

def main():
    tf.reset_default_graph
    input_node = tf.placeholder(tf.float32,shape=[1,64,64,3],name='input_image')
    f_preds = model.model(input_node,is_training=False)
#     import pdb
#     pdb.set_trace()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,model_path)
        tf.train.write_graph(sess.graph_def,'./','slim.pb')
        freeze_graph.freeze_graph('./slim.pb','',False,model_path,'logits/pred/Relu','save/restore_all', 'save/Const:0','./slim_Landmark.pb',False,'','')
        print('done')

if __name__ == '__main__':
    main()