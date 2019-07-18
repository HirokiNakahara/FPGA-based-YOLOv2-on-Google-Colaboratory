import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

#from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image

import random
import cv2

import argparse
import glob

parser = argparse.ArgumentParser(description='Generate text file for ground truth dataset')
parser.add_argument('--label_file', '-l', type=str, default='voc', help='dataset label')
parser.add_argument('--data_path', '-d', type=str, default='none', help='dataset path')
parser.add_argument('--out_path', '-o', type=str, default='none', help='output path')
parser.add_argument('--img_size', '-s', type=int, default=213, help='test image size')

#parser.add_argument('--label_file', '-l', type=str, default='voc', help='CLASS LABEL FILE PATH')

'''
$ python gen_ground_truth.py -d /home/nakahara/dataset/TrainingDataset/VOC2012/VOCdevkit/VOC2012 \
          -l voc3 (or voc3_label.txt) -o mAP/VOC_ground_truth -s 213
'''
######################################################################
args = parser.parse_args()

#data_dir = '/home/nakahara/dataset/VOC2012/VOCdevkit/VOC2012/'
#output_dir = './mAP/VOC_ground_truth/'

data_dir = args.data_path
output_dir = args.out_path

files = glob.glob(data_dir + '/Annotations/' + '/*.xml')

ids = []
for i in range(len(files)):
    id_name = files[i]
    id_name = id_name[:-4]
    id_name = id_name.rsplit('/',1)
    ids.append(id_name[1])

print("[INFO] #ANNOTATIONS %d" % len(ids))
#print(ids)
#exit()

if args.label_file == 'voc3':
    label_names=('car','person','bicycle','other')
elif args.label_file == 'voc':
    label_names=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor') # VOC original
else:
    label_names = open(args.label_file).read().split()
    print("[INFO] LABEL FILE %s" % args.label_file)

datasize = len(ids)
#datasize = 10
#print("size=%d" % datasize)

selected_list = ""
print("")

ratio = 3

for idx_data in range(datasize):
#for idx_data in range(40,140):
#    print("%d/%d" % (idx_data,datasize))
#    print("\033[1A%d/%d" % (idx_data,datasize))

    registered = 0
#    while registered == 0:
#    idx_data = random.randint(0,datasize-1)

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

    # load an image
    img_file = os.path.join(data_dir, 'JPEGImages', id_ + '.jpg')
    img = read_image(img_file, color=True)
    ch, h, w = img.shape
    test_img = cv2.imread(img_file)
    test_img = cv2.resize(test_img, (int(args.img_size*ratio),int(args.img_size*ratio)))


    for obj in anno.findall('object'):
    # when in not using difficult split, and the object is
    # difficult, skipt it.
#    if not self.use_difficult and int(obj.find('difficult').text) == 1:
#    continue

#        print(obj)

        name = obj.find('name').text.lower().strip()

#    label_names=('car','person','bicycle','other')
#    label_names=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor') # VOC original
        if name == "motorbike":
            name = "bicycle"
        if name == "bus":
            name = "car"

#        print("name,",name)
        if(name not in label_names):
#            print("label(other),",label_names)
            name = 'other'
#            continue

        label.append(label_names.index(name))
        difficult.append(int(obj.find('difficult').text))
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based

        ymin = int(float(bndbox_anno.find('ymin').text)) - 1
        xmin = int(float(bndbox_anno.find('xmin').text)) - 1
        ymax = int(float(bndbox_anno.find('ymax').text)) - 1
        xmax = int(float(bndbox_anno.find('xmax').text)) - 1

        # Load an image
#        print("load %s" % (self.data_dir + 'JPEGImages' + id_ + '.jpg'))
#        img_file = os.path.join(data_dir, 'JPEGImages', id_ + '.jpg')
#        img = read_image(img_file, color=True)
#        ch, h, w = img.shape

#        test_img = cv2.imread(img_file)
#        h, w, ch = img.shape[:3]
#        print(img.shape)

#        print("org w=%d,h=%d xmin=%d ymin=%d xmax=%d ymax=%d" % (w,h,xmin,ymin,xmax,ymax))
        # resize image to adjust 1:1 aspect ratio
        if args.img_size < w:
            xmin = int(xmin * (args.img_size / w))
            xmax = int(xmax * (args.img_size / w))
        else:
            xmin = int(xmin * (w / args.img_size))
            xmax = int(xmax * (w / args.img_size))

        if args.img_size < h:
            ymin = int(ymin * (args.img_size / h))
            ymax = int(ymax * (args.img_size / h))
        else:
            ymin = int(ymin * (h / args.img_size))
            ymax = int(ymax * (h / args.img_size))

        # check area for BBOX
        area_ratio = float((xmax - xmin) * (ymax - ymin)) / float(args.img_size ** 2)
#
#        if area_ratio < 0.05:
#            continue

#        print("resized w=%d,h=%d xmin=%d ymin=%d xmax=%d ymax=%d" % (w,h,xmin,ymin,xmax,ymax))        
        line = "%s %d %d %d %d\n" % (name,xmin,ymin,xmax,ymax)
        gts += line

        # draw bounding box (for debug)
#        cv2.rectangle( test_img, (xmin,ymin),(xmax,ymax),(0,255,0), 1)
        if name == 'car':
            score_txt = 'car:' + str(area_ratio)
            color = (0,255,0)
        elif name == 'person':
            score_txt = 'person:' + str(area_ratio)
            color = (0,0,255)
        elif name == 'bicycle':
            score_txt = 'bicycle:' + str(area_ratio)
            color = (255,0,0)
        else:
            score_txt = 'other:' + str(area_ratio)
            color = (255,0,255)

        cv2.rectangle( test_img, (xmin*ratio,ymin*ratio),(xmax*ratio,ymax*ratio),color, 3)
        cv2.rectangle( test_img, (xmin*ratio,ymin*ratio-30),(xmax*ratio,ymin*ratio),color, -1)
        cv2.putText( test_img, score_txt, (xmin*ratio,ymin*ratio-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

    path = os.path.join(output_dir,id_ + '.txt')
#    print(path)
#    print(gts)
    with open(path, mode='w') as f:
        f.write(gts)

    # draw bounding box (for debug)
#    cv2.imshow("test image", test_img)
#    cv2.waitKey(0)


print("JOB COMPLETE") 
