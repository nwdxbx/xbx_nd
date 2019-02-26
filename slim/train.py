import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import model
from data_generator.data_pt import TrainDataGenerator

tf.app.flags.DEFINE_integer("input_size", 64, "")
tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("kp", 7, "")
tf.app.flags.DEFINE_float("learning_rate",0.01,"")
tf.app.flags.DEFINE_integer("max_steps", 551500,"")
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 5515, '')
tf.app.flags.DEFINE_integer('val_steps', 5515, '')
tf.app.flags.DEFINE_float('maxVal_loss', 100., '')
tf.app.flags.DEFINE_float("moving_average_decay", 0.997, "")
tf.app.flags.DEFINE_string("checkpoint_path", "log/666/", "")
tf.app.flags.DEFINE_string("annotation_path", "./300VW_anos.txt", "")
tf.app.flags.DEFINE_string("pretrained_model_path", None, "")
tf.app.flags.DEFINE_boolean("restore", False, "whether to resotre from checkpoint")

FLAGS = tf.app.flags.FLAGS

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    input_images = tf.placeholder(tf.float32,shape=[None,64,64,3],name="input_images")
    preds = tf.placeholder(tf.float32,shape=[None,14],name="pred_offset")
    data_gen = TrainDataGenerator(FLAGS.annotation_path,FLAGS.batch_size,FLAGS.kp)

    # import pdb
    # pdb.set_trace()
    FLAGS.max_steps = data_gen.train_step*100
    FLAGS.save_checkpoint_steps = data_gen.train_step
    FLAGS.val_steps = data_gen.val_step
    
    global_step = tf.get_variable("global_step",[],initializer=tf.constant_initializer(0),trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,decay_steps=55160,decay_rate=0.6)
    opt = tf.train.AdamOptimizer(learning_rate)

    with tf.name_scope("model") as scope:
        f_preds = model.model(input_images,is_training=True)
        loss = tf.reduce_mean(tf.abs(f_preds-preds))*100
        tf.contrib.quantize.create_training_graph(quant_delay=0)
        total_loss = tf.add_n([loss]+tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        grads = opt.compute_gradients(loss)
    
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
    summary_op = tf.summary.merge_all()
    
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name="train_op")
    
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
    init = tf.global_variables_initializer()
    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        print("model loaded sucessfully")
        #tf.train.write_graph(sess.graph_def,"./","landmark.pbtxt",as_text=False)
        start = time.time()
        train_generator = data_gen.get_train()
        val_generator = data_gen.get_val()
        for step in range(1,FLAGS.max_steps):    
            train_data = next(train_generator)
            lr ,ml ,tl, _ = sess.run([learning_rate,loss,total_loss,train_op],feed_dict={input_images: train_data[0],
                                                                        preds: train_data[1]})                                                        
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break
            if step % 10 == 0:
                avg_time_per_step = (time.time()-start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size)/(time.time() - start)
                start = time.time()
                print("step {:06d}, model loss {:.4f}, total loss {:.4f}, learning rate {:.8f}, {:.2f} seconds/step, {:.2f} examples/second".format(
                    step,ml,tl,lr,avg_time_per_step,avg_examples_per_second
                ))

            if step % FLAGS.save_checkpoint_steps == 0:
                val_loss = 0
                for val_step in range(FLAGS.val_steps):
                    val_data = next(val_generator)
                    lr,ml = sess.run([learning_rate,loss],feed_dict={input_images:val_data[0],
                                                                preds:val_data[1]})
                    val_loss = val_loss + ml
                    print("val_step {:06d}, model loss {:.4f}, learning rate {:.8f}".format(val_step,ml,lr))
                avgval_loss = val_loss/FLAGS.val_steps
                print ("avg loss {:.4f}".format(avgval_loss))
                if(avgval_loss<FLAGS.maxVal_loss):
                    FLAGS.maxVal_loss=avgval_loss
                    print("saving model...")
                    saver.save(sess,FLAGS.checkpoint_path+ "val_loss_{}_model.ckpt".format(avgval_loss),global_step=global_step)

if __name__ == "__main__":
    main()
