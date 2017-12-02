import tensorflow as tf
import tensorflow.contrib.slim as slim

class AnimalRecognition:

    def __init__(self, cls, img_size, img_channels, model_dir):
        self._classes = cls
        self._img_size = img_size
        self._num_channels = img_channels
        self._model_dir = model_dir
        tf.set_random_seed(2)
        self._session = tf.Session()
        self._images = tf.placeholder(tf.float32, shape=[None, self._img_size,self._img_size,self._num_channels], name='images')
        self._labels = tf.placeholder(tf.float32, shape=[None, len(self._classes)], name='labels')
        self._labels_cls = tf.argmax(self._labels, dimension=1)


    def create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    def create_convolutional_layer(self, input,
                                   num_input_channels,
                                   conv_filter_size,
                                   num_filters):
        ## We shall define the weights that will be trained using create_weights function.
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        ## We create biases using the create_biases function. These are also trained.
        biases = self.create_biases(num_filters)

        ## Creating the convolutional layer
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        layer += biases

        ## We shall be using max-pooling.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        ## Output of pooling is fed to Relu which is the activation function for us.
        layer = tf.nn.relu(layer)
        return layer

    # The Output of a convolutional layer is a multi-dimensional Tensor.
    # We want to convert this into a one-dimensional tensor.
    def create_flatten_layer(self, layer):
        # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()

        ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements()

        ## Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features])

        return layer

    # fully connected layer
    def create_fc_layer(self, input,
                        num_inputs,
                        num_outputs,
                        use_relu=True):
        # Let's define trainable weights and biases.
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)

        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def createCNNNetwork(self):

        network = self.create_convolutional_layer(input=self._images,
                                                 num_input_channels=self._num_channels,
                                                 conv_filter_size=3,
                                                 num_filters=32)

        network = self.create_convolutional_layer(input=network,
                                                   num_input_channels=32,
                                                   conv_filter_size=3,
                                                   num_filters=32)

        network = self.create_convolutional_layer(input=network,
                                                   num_input_channels=32,
                                                   conv_filter_size=3,
                                                   num_filters=64)

        network = self.create_flatten_layer(network)

        network = self.create_fc_layer(input=network,
                                             num_inputs= network.get_shape()[1:4].num_elements(),
                                             num_outputs=128)

        network = self.create_fc_layer(input=network,
                                             num_inputs= 128,
                                             num_outputs=len(self._classes),use_relu=False)

        # build predicts
        prd_ary = tf.nn.softmax(network,name='animal_prd')
        prd_clses = tf.argmax(prd_ary, dimension=1)

        # build cost
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=network,
                                                                labels= self._labels)
        self._cost = tf.reduce_mean(cross_entropy)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._cost)

        correct_prediction = tf.equal(prd_clses, self._labels_cls)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._session.run(tf.global_variables_initializer())

    def createCNNNetworkWithSlim(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.05),
                            weights_regularizer=slim.l2_regularizer(0.05)):
            net = slim.conv2d(self._images, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 128, scope='fc1')
            net = slim.fully_connected(net, len(self._classes), activation_fn=None, scope='fc2')
            return net

    def createPrdAndCost(self):
        network = self.createCNNNetworkWithSlim()
        # build predicts
        prd_ary = tf.nn.softmax(network,name='animal_prd')
        prd_clses = tf.argmax(prd_ary, dimension=1)

        # build cost
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=network,
                                                                labels= self._labels)
        self._cost = tf.reduce_mean(cross_entropy)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._cost)

        correct_prediction = tf.equal(prd_clses, self._labels_cls)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._session.run(tf.global_variables_initializer())


    def train(self, num_iteration, dataTrainBatch, dataValBatch):
        #self.createCNNNetwork()
        self.createPrdAndCost()
        '''
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))
        '''
        saver = tf.train.Saver()
        for i in range(num_iteration):

            for j in range (dataTrainBatch.getBatchSize()):
                batch_img_tr, batch_labels_tr = dataTrainBatch.getBatchAtIndex(j)
                feed_dict_train = {self._images: batch_img_tr, self._labels: batch_labels_tr}
                self._session.run(self._optimizer, feed_dict=feed_dict_train)

            print ('Epoch '+str(i)+' finished')

            if i % 20 == 0:
                batch_img_val, batch_labels_val = dataValBatch.getNextBatch()
                feed_dict_val = {self._images: batch_img_val, self._labels: batch_labels_val}
                val_loss = self._session.run(self._cost, feed_dict=feed_dict_val)
                self.show_progress(i, feed_dict_train, feed_dict_val, val_loss)
                saver.save(self._session, save_path = self._model_dir+'/animal-model')
                tf.train.write_graph(self._session.graph_def, self._model_dir, 'animal-model.pbtxt')

    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        acc = self._session.run(self._accuracy, feed_dict=feed_dict_train)
        val_acc = self._session.run(self._accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        print(msg.format(epoch, acc, val_acc, val_loss))