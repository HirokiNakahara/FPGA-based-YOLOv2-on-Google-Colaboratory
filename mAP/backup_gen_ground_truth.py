import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

#from chainercv.datasets.voc import voc_utils
#from chainercv.utils import read_image

import random
#import cv2

data_dir = '/home/nakahara/dataset/VOC2012/VOCdevkit/VOC2012/'
id_list_file = '../dataset_id_list.txt'
output_dir = './VOC_ground_truth/'

ids = [id_.strip() for id_ in open(id_list_file)]

data_dir = data_dir
#class_names=('car','person','bicycle')
label_names=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor') # VOC original

datasize = 10 #len(ids)
#    print("size=%d" % datasize)

selected_list = ""

for idx_data in range(datasize):
#    print("%d/%d" % (idx_data,datasize))
    registered = 0
#    while registered == 0:
    idx_data = random.randint(0,datasize-1)

    id_ = ids[idx_data]
    anno = ET.parse(
    os.path.join(data_dir, 'Annotations', id_ + '.xml'))
    bbox = list()
    label = list()
    difficult = list()

    # Load a image
#    print("load %s" % (self.data_dir + 'JPEGImages' + id_ + '.jpg'))
#    img_file = os.path.join(data_dir, 'JPEGImages', id_ + '.jpg')
#    img = read_image(img_file, color=True)
#    tmp_img = cv2.imread(img_file)
#    h, w, ch = tmp_img.shape[:3]
#    img_area = h*w

    n_skips = 0

    gts = ''

    for obj in anno.findall('object'):
    # when in not using difficult split, and the object is
    # difficult, skipt it.

#    if not self.use_difficult and int(obj.find('difficult').text) == 1:
#    continue

#        print(obj)

        name = obj.find('name').text.lower().strip()

        label.append(label_names.index(name))
        difficult.append(int(obj.find('difficult').text))
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based

        ymin = int(float(bndbox_anno.find('ymin').text)) - 1
        xmin = int(float(bndbox_anno.find('xmin').text)) - 1
        ymax = int(float(bndbox_anno.find('ymax').text)) - 1
        xmax = int(float(bndbox_anno.find('xmax').text)) - 1

        # compute ratio for area to be eliminated
#        bbox_area = (xmax-xmin)*(ymax-ymin)
#        area_ratio = bbox_area / img_area
#        aspect_ratio = (xmax-xmin) / (ymax-ymin)

#        print("img=%d bbox=%d ratio(area)=%f (aspect)=%f" % (img_area,bbox_area,area_ratio,aspect_ratio))

        # register bbox
#        bbox.append([
#            int(float(bndbox_anno.find(tag).text)) - 1
#            for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        line = "%s %d %d %d %d\n" % (name,xmin,ymin,xmax,ymax)
        gts += line

    path = os.path.join(output_dir,id_ + '.txt')
    print(path)
    print(gts)
    with open(path, mode='w') as f:
        f.write(gts)