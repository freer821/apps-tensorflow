from libs.data_handler import get_image_array_from_file, get_labels_array
import tensorflow as tf
import numpy as np
import os

class TestModel:

    def __init__(self, model_dir, label_path):
        self.model_dir = model_dir
        self._label_path = label_path
        self.initMetaUndPDName()

    def initGraph_with_meta(self):
        self._sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(self.model_dir, self._metaName))
        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(self._sess, tf.train.latest_checkpoint(self.model_dir))
        self._graph = tf.get_default_graph()

    def initGraph_with_pd(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(os.path.join(self.model_dir, self._pdName), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self._sess = tf.Session(graph=self._graph)

    def initMetaUndPDName(self):
        self._metaName  = ''
        self._pdName   = ''
        for file in os.listdir(self.model_dir):
            if len(file) > 5 and file[-5:] == '.meta':
                self._metaName = file
            if len(file) > 3 and file[-3:] == '.pb':
                self._pdName = file

class TestEmotionModel(TestModel):

    def __init__(self, model_path, label_path):
        TestModel.__init__(self,model_path, label_path)

    def predict(self, img_path):
        images = []
        image = get_image_array_from_file(img_path, 48)
        images.append(image)
        labels = np.zeros((1,len(get_labels_array(self._label_path))))

        '''
        for op in self._graph.get_operations():
            print(str(op.name))

        '''

        # output placeholders
        tf_emo = self._graph.get_tensor_by_name("emo_prd:0")

        # input placeholders
        tf_imgs = self._graph.get_tensor_by_name("images:0")
        #tf_labels = self._graph.get_tensor_by_name("labels:0")

        feed_dict_test = {tf_imgs: images}

        result = self._sess.run(tf_emo, feed_dict=feed_dict_test)
        print(result)
        print(labels)


class TestAnimalModel(TestModel):

    def __init__(self, model_path, label_path):
        TestModel.__init__(self,model_path, label_path)

    def predict(self, img_path):
        images = []
        image = get_image_array_from_file(img_path, 128)
        images.append(image)
        labels = np.zeros((1,len(get_labels_array(self._label_path))))

        '''
        for op in self._graph.get_operations():
            print(str(op.name))

        '''

        # output placeholders
        tf_animial = self._graph.get_tensor_by_name("animal_prd:0")

        # input placeholders
        tf_imgs = self._graph.get_tensor_by_name("images:0")
        #tf_labels = self._graph.get_tensor_by_name("labels:0")

        feed_dict_test = {tf_imgs: images}

        result = self._sess.run(tf_animial, feed_dict=feed_dict_test)
        print(result)
