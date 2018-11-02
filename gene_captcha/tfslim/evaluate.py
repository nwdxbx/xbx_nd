import numpy as np
import tensorflow as tf

from captcha.image import ImageCaptcha

flags = tf.app.flags
flags.DEFINE_string('model_ckpt_path', None, 'Path to model checkpoint.')
FLAGS = flags.FLAGS


def generate_captcha(text='1'):
    capt = ImageCaptcha(width=28, height=28, font_sizes=[24])
    image = capt.generate_image(text)
    image = np.array(image, dtype=np.uint8)
    return image


def main(_):
    with tf.Session() as sess:
        ckpt_path = FLAGS.model_ckpt_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        classes = tf.get_default_graph().get_tensor_by_name('classes:0')
        
        for i in range(10):
            label = np.random.randint(0, 10)
            image = generate_captcha(str(label))
            image_np = np.expand_dims(image, axis=0)
            predicted_label = sess.run(classes, 
                                       feed_dict={inputs: image_np})
            print(predicted_label, ' vs ', label)
            
            
if __name__ == '__main__':
    tf.app.run()