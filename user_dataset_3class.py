import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

import chainer

from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image

import cv2
import glob

class UserDataset3Class(chainer.dataset.DatasetMixin):

    def __init__(self, anno_dir='auto', img_dir='auto', cls_label='auto', split='train', year='2012',
                 use_difficult=False, return_difficult=False):

#        data_dir = '/home/nakahara/dataset/TrainingDataset/VOC2012/VOCdevkit/VOC2012/'
#        id_list_file = './test_id_list.txt'

#        print(anno_dir,img_dir,cls_label)

        files = glob.glob(anno_dir + '/*.xml')
#        print(files)
#        print(len(files))

        ids = []
        for i in range(len(files)):
            id_name = files[i]
            id_name = id_name[:-4]
#            print("%d %s" % (i,id_name))
            id_name = id_name.rsplit('/',1)
#            print(id_name[1])
            ids.append(id_name[1])
#        exit()

#        self.ids = [id_.strip() for id_ in open(id_list_file)]
#        self.anno_ids = files
        self.ids = ids
#        print(self.ids)
#        print(len(self.ids))
#        exit()
        self.anno_dir = anno_dir
        self.img_dir = img_dir

        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
#        self.class_names=cls_label #('car','person','bicycle')
#        print(type(self.class_names))
#        self.class_names=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
#        print(type(self.class_names))
        self.class_names = tuple(cls_label)

#        print(self.class_names)
#        exit()

        self.candidate_class_names = ('bicycle', 'bus', 'car', 'motorbike', 'person')

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        datasize = len(self.ids)
#        print("size=%d" % datasize)

        id_ = self.ids[i]
        anno = ET.parse(
#            os.path.join(self.anno_dir, 'Annotations', id_ + '.xml'))
            os.path.join(self.anno_dir, id_ + '.xml'))
#        anno = id_
#        print(anno)

#        print(" -->> load %s" % ("Annotations" + str(id_)+'.xml' ))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.

#            if not self.use_difficult and int(obj.find('difficult').text) == 1:
#                continue
#            print(obj)

            name = obj.find('name').text.lower().strip()

#            print(name)
#            if(name not in self.class_names):
            if(name not in self.candidate_class_names):
                name = 'other'
#                continue

#            print("append %s" % name)
            if name == 'motorbike':
                name = 'bicycle'
            if name == 'bus':
                name = 'car'

#            print(name)

            label.append(self.class_names.index(name))
#            label.append(name)

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(float(bndbox_anno.find(tag).text)) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
#            print(bbox)
#        print("extract bboxs")
#        print(bbox)
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)

        # Load a image
#        img_file = os.path.join(self.img_dir, 'JPEGImages', id_ + '.jpg')
        img_file = os.path.join(self.img_dir, id_ + '.jpg')
#        img_file = self.img_ids[i]
        img = read_image(img_file, color=True)

#        print(img)
#        print(img.shape)
#        cv2.imwrite("tmp.jpg", img.transpose(1,2,0).astype(np.uint8))
#        exit(0)

        if self.return_difficult:
            return img, bbox, label, difficult

#        print(img)
#        print(bbox)
#        print(label)
        return img, bbox, label
