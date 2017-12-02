import os
import csv
import cv2
from itertools import islice
import numpy as np
from enum import Enum
from PIL import Image
from sklearn.utils import shuffle

def get_image_array_from_file(infilename, img_size):
    image = cv2.imread(infilename)
    image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    return image

def convert_str_to_image(image_blob, img_size):
    ''' Convert a string blob to an image object. '''
    image_string = image_blob.split(' ')
    image_array = np.asarray(image_string, dtype=np.float32).reshape(img_size, img_size)
    image = Image.fromarray(image_array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def get_labels_array(file_path):
    with open(file_path) as f:
        content = [x.strip('\n') for x in f.readlines()]
        return content
    return []

class DataType(Enum):
    Train = 'train'
    Test  = 'test'
    Val   = 'val'

class DataCreator:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.checkDIR(self.base_dir)

    def checkDIR(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        return directory

    def create_images_from_csv(self, file_path, img_size):

        if not os.path.exists(file_path):
            print (file_path+' not found!')
            return

        print ('begin to create images from cvs: '+file_path)
        index = 0
        with open(file_path, 'r') as csvfile:
            ferplus_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                image_path = os.path.join(self.checkDIR(self.base_dir+'/'+row[2]+'/'+row[0]),'img_'+str(index)+'.jpg')
                image = convert_str_to_image(row[1], img_size)
                image.save(image_path, compress_level=0)
                index+=1

        print('creating images finished')


class DataLoader:

    def __init__(self, cls_path, img_size, img_channels):
        self._img_size = img_size
        self._img_channels=img_channels
        self.initTestDataSets()
        self.initTrainDataSets()
        self.initValDataSets()
        self._classes = get_labels_array(cls_path)

    def initTrainDataSets(self):
        self._tr_images = []
        self._tr_labels = []

    def initValDataSets(self):
        self._val_images = []
        self._val_labels = []

    def initTestDataSets(self):
        self._test_images = []
        self._test_labels = []

    def addImageAndLabel(self, img_array, cls_index, data_typ):
        label = np.zeros(len(self._classes))
        label[cls_index]=1.0

        if data_typ == DataType.Train:
            self._tr_images.append(img_array)
            self._tr_labels.append(label)
        elif data_typ == DataType.Test:
            self._test_images.append(img_array)
            self._test_labels.append(label)
        elif data_typ == DataType.Val:
            self._val_images.append(img_array)
            self._val_labels.append(label)
        else:
            print ('no data type found!')

    def load_images_from_dir(self, dir_path, data_typ='train'):

        if not os.path.isdir(dir_path):
            print("sorry, path not found: "+ dir_path)
        else:
            for clsname in os.listdir(dir_path):
                if clsname in self._classes:
                    print("begin to load images from " + dir_path + "/"+clsname)
                    for filename in os.listdir(dir_path + "/"+clsname):
                        image_array = get_image_array_from_file(dir_path + "/"+clsname + '/' + filename, self._img_size)
                        self.addImageAndLabel(image_array, self._classes.index(clsname), data_typ)
                    print("load images finished!")

    def get_tr_images(self):
        return np.array(self._tr_images)

    def get_tr_labels(self):
        return np.array(self._tr_labels)

    def get_test_images(self):
        return np.array(self._test_images)

    def get_test_labels(self):
        return np.array(self._test_labels)

    def get_val_images(self):
        return np.array(self._val_images)

    def get_val_labels(self):
        return np.array(self._val_labels)

    def get_classes(self):
        return self._classes

    def get_img_size(self):
        return self._img_size

    def get_img_channels(self):
        return self._img_channels


class DataBatch:
    def __init__(self, imgs, labels, batch_size):
        self._imgs,self._labels=shuffle(imgs,labels)
        self._length = len(imgs)
        self._batch_size = batch_size
        self._batch_index = 0

    def getBatchAtIndex(self, index):
        batch_imgs = []
        batch_labels = []
        startIndex = index*self._batch_size
        if (startIndex + self._batch_size)  < self._length:
            batch_imgs = self._imgs[startIndex: startIndex+ self._batch_size]
            batch_labels = self._labels[startIndex: startIndex+ self._batch_size]
        else:
            batch_imgs = self._imgs[startIndex: self._length]
            batch_labels = self._labels[startIndex: self._length]

        return batch_imgs, batch_labels

    def getBatchSize(self):
        return int(round(float(self._length) / self._batch_size))

    def getNextBatch(self):
        batch_imgs = []
        batch_labels = []

        if self._batch_index <  self.getBatchSize():
            batch_imgs, batch_labels = self.getBatchAtIndex(self._batch_index)
            self._batch_index += 1
        else:
            self._batch_index = 0
            batch_imgs, batch_labels = self.getBatchAtIndex(self._batch_index)

        return batch_imgs, batch_labels
