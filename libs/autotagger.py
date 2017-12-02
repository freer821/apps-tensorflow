import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os

class AutoTagger:

    def __init__(self, img_np, base_path, folder_name, img_name, img_channels = 3):
        self._img = img_np
        self._base_path = base_path
        self._folder_name = folder_name
        self._img_name = img_name
        self._img_width = img_np.shape[1]
        self._img_height = img_np.shape[0]
        self._img_channels = img_channels
        self._objs = []
        self._rootElment = ET.Element('annotation')
        self._folder_path = os.path.join(self._base_path,self._folder_name)
        self.checkDir(self._base_path)
        self.checkDir(self._folder_path)

    def checkDir(self, dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    def addObject(self, cls, box):
        self._objs.append({'classname': cls, 'box': box})

    def save(self):
        plt.imsave(self._folder_path + '/' + self._img_name, self._img)
        self.buildXML()
        xmlTree = ET.ElementTree(self._rootElment)
        filename, file_extension = os.path.splitext(self._img_name)
        xmlTree.write(self._folder_path + '/' +filename+'.xml')

    def buildXML(self):
        self.buildSimpleElement('folder', self._folder_name)
        self.buildSimpleElement('filename', self._img_name)
        self.buildSimpleElement('path', os.path.join(self._folder_path, self._img_name))
        self.buildSource()
        self.buildSize()
        self.buildSimpleElement('segmented', '0')
        for obj in self._objs:
            self.buildObject(obj)

    def buildSimpleElement(self, elmentName, elmentVal = None):
        el = ET.SubElement(self._rootElment, elmentName)
        if elmentVal is not None:
            el.text = elmentVal
        return el

    def buildSource(self, database='Unknown'):
        source = ET.SubElement(self._rootElment, 'source')
        db = ET.SubElement(source, 'database')
        db.text = database

    def buildSize(self):
        size = ET.SubElement(self._rootElment, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(self._img_width)
        height = ET.SubElement(size, 'height')
        height.text = str(self._img_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(self._img_channels)

    def buildObject(self, obj):
        object = ET.SubElement(self._rootElment, 'object')
        name = ET.SubElement(object, 'name')
        name.text = obj['classname']
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(obj['box'][1] * self._img_height)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(obj['box'][0] * self._img_width)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(obj['box'][3] * self._img_height)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(obj['box'][2]* self._img_width)