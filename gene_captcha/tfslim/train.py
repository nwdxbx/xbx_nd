import cv2
import glob
import numpy as np
import os
import tensorflow as tf

import model

flags = tf.app.flags

flags.DEFINE_string('images_path', None, 'Path to training images.')
flags.DEFINE_string('model_output_path', None, 'Path to model checkpoint.')
FLAGS = flags.FLAGS


def get_train_data(images_path):
    """Get the training images from images_path.
    
    Args: 
        images_path: Path to trianing images.
        
    Returns:
        images: A list of images.
        lables: A list of integers representing the classes of images.
        
    Raises:
        ValueError: If images_path is not exist.
    """
    if not os.path.exists(images_path):
        raise ValueError('images_path is not exist.')
        
    images = []
    labels = []
    images_path = os.path.join(images_path, '*.jpg')
    count = 0
    for image_file in glob.glob(images_path):
        count += 1
        if count % 100 == 0:
            print('Load {} images.'.format(count))
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Assume the name of each image is imagexxx_label.jpg
        label = int(image_file.split('_')[-1].split('.')[0])
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def next_batch_set(images, labels, batch_size=128):
    """Generate a batch training data.
    
    Args:
        images: A 4-D array representing the training images.
        labels: A 1-D array representing the classes of images.
        batch_size: An integer.
        
    Return:
        batch_images: A batch of images.
        batch_labels: A batch of labels.
    """
    indices = np.random.choice(len(images), batch_size)
    batch_images = images[indices]
    batch_labels = labels[indices]
    return batch_images, batch_labels


def main(_):
    inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    
    cls_model = model.Model(is_training=True, num_classes=10)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    classes_ = tf.identity(classes, name='classes')
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step)
    
    saver = tf.train.Saver()
    
    images, targets = get_train_data(FLAGS.images_path)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(6000):
            batch_images, batch_labels = next_batch_set(images, targets)
            train_dict = {inputs: batch_images, labels: batch_labels}
            
            sess.run(train_step, feed_dict=train_dict)
            
            loss_, acc_ = sess.run([loss, acc], feed_dict=train_dict)
            
            train_text = 'step: {}, loss: {}, acc: {}'.format(
                i+1, loss_, acc_)
            print(train_text)
            
        saver.save(sess, FLAGS.model_output_path)
        
    
if __name__ == '__main__':
    tf.app.run()
