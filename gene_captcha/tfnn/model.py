import tensorflow as tf

        
class Model():
    """A simple 10-classification CNN model definition."""
    
    def __init__(self,
                 is_training,
                 num_classes):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        self._num_classes = num_classes
        self._is_training = is_training
        
    def preprocess(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        shape = preprocessed_inputs.get_shape().as_list()
        height, width, num_channels = shape[1:]

        conv1_weights = tf.get_variable(
            'conv1_weights', shape=[3, 3, num_channels, 32], 
            dtype=tf.float32)
        conv1_biases = tf.get_variable(
            'conv1_biases', shape=[32], dtype=tf.float32)
        conv2_weights = tf.get_variable(
            'conv2_weights', shape=[3, 3, 32, 32], dtype=tf.float32)
        conv2_biases = tf.get_variable(
            'conv2_biases', shape=[32], dtype=tf.float32)
        conv3_weights = tf.get_variable(
            'conv3_weights', shape=[3, 3, 32, 64], dtype=tf.float32)
        conv3_biases = tf.get_variable(
            'conv3_biases', shape=[64], dtype=tf.float32)
        conv4_weights = tf.get_variable(
            'conv4_weights', shape=[3, 3, 64, 64], dtype=tf.float32)
        conv4_biases = tf.get_variable(
            'conv4_biases', shape=[64], dtype=tf.float32)
        conv5_weights = tf.get_variable(
            'conv5_weights', shape=[3, 3, 64, 128], dtype=tf.float32)
        conv5_biases = tf.get_variable(
            'conv5_biases', shape=[128], dtype=tf.float32)
        conv6_weights = tf.get_variable(
            'conv6_weights', shape=[3, 3, 128, 128], dtype=tf.float32)
        conv6_biases = tf.get_variable(
            'conv6_biases', shape=[128], dtype=tf.float32)
            
        flat_height = height // 4
        flat_width = width // 4
        flat_size = flat_height * flat_width * 128
        
        fc7_weights = tf.get_variable(
            'fc7_weights', shape=[flat_size, 512], dtype=tf.float32)
        fc7_biases = tf.get_variable(
            'f7_biases', shape=[512], dtype=tf.float32)
        fc8_weights = tf.get_variable(
            'fc8_weights', shape=[512, 512], dtype=tf.float32)
        fc8_biases = tf.get_variable(
            'f8_biases', shape=[512], dtype=tf.float32)
        fc9_weights = tf.get_variable(
            'fc9_weights', shape=[512, self._num_classes], dtype=tf.float32)
        fc9_biases = tf.get_variable(
            'f9_biases', shape=[self._num_classes], dtype=tf.float32)
        
        net = preprocessed_inputs
        net = tf.nn.conv2d(net, conv1_weights, strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
        net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv2_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        net = tf.nn.conv2d(net, conv3_weights, strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv3_biases))
        net = tf.nn.conv2d(net, conv4_weights, strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv4_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        net = tf.nn.conv2d(net, conv5_weights, strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv5_biases))
        net = tf.nn.conv2d(net, conv6_weights, strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv6_biases))
        net = tf.reshape(net, shape=[-1, flat_size])
        net = tf.nn.relu(tf.add(tf.matmul(net, fc7_weights), fc7_biases))
        net = tf.nn.relu(tf.add(tf.matmul(net, fc8_weights), fc8_biases))
        net = tf.add(tf.matmul(net, fc9_weights), fc9_biases)
        prediction_dict = {'logits': net}
        return prediction_dict
    
    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        postprecessed_dict = {'classes': classes}
        return postprecessed_dict
    
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        logits = prediction_dict['logits']
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=groundtruth_lists))
        loss_dict = {'loss': loss}
        return loss_dict
