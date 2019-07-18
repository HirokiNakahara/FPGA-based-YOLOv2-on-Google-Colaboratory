import numpy as np
import chainer
import os
from chainer import serializers, optimizers, cuda, training
from chainer.training import extension,extensions,updaters
#from model import YOLOv2Predictor,AlexYOLOv2,MobileYOLOv2,TernaryAlexYOLOv2, GUINNESS_YOLOv2
#from model import YOLOv2Predictor,GUINNESS_YOLOv2, MobileYOLOv2
#from lib.image_generator import *

from guinness_net_yolov2 import GUINNESS_YOLOv2
from yolo_predictor import YOLOv2Predictor

from lib.npz import load_npz
import argparse
import datetime
from matplotlib import pylab as plt

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainer.datasets import TransformDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv import transforms
from chainercv.links.model.ssd import random_crop_with_bbox_constraints
#from datasets import ConcatenatedDataset,CarPersonBboxDataset,VOC3BboxDataset #, sg_label_names
from chainer.dataset import to_device
from chainer.dataset.convert import _concat_arrays
from chainercv.visualizations import vis_bbox
from matplotlib import pylab as plt
from chainercv.evaluations import eval_detection_voc
#from datasets import ConcatenatedDataset #,SgBboxDataset

import chainer.links as L
import chainer.functions as F
#from lab import functions as LF
#from lab import links as LL
import cv2

#from lab.functions import relu6
from chainer import Link, Chain, ChainList
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, initializers, reporter

import glob
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Detection Results')
    parser.add_argument('--img', '-i', type=str, default='test.jpg')
    parser.add_argument('--img_size', '-s', type=int, default=181, help='test image size')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID (negative value uses CPU)')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Initial learning rate for Optimizer')

    parser.add_argument('--label_file', '-l', type=str, default='voc', help='dataset label')
    parser.add_argument('--data_path', '-d', type=str, default='none', help='dataset path')
    parser.add_argument('--out_path', '-o', type=str, default='none', help='output path')

    '''
    # python gen_detection_results.py -s 213 -g 0 --pretrained_model ./logs/emsemble_yolov2/dense_yolov2_1/mymodel.npz \
                       -l voc3 -d /home/nakahara/dataset/TrainingDataset/VOC2012/VOCdevkit/VOC2012 \
                       -o mAP/VOC_detection_results
    '''

    args = parser.parse_args()

#    data_dir = '/home/nakahara/dataset/VOC2012/VOCdevkit/VOC2012/'
#    output_dir = './mAP/VOC_detection_results/'
    data_dir = args.data_path
#    id_list_file = args.list
    output_dir = args.out_path

    if args.label_file == 'voc3':
        label_names=('car','person','bicycle','other')
    elif args.label_file == 'voc':
        label_names=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor') # VOC original
    else:
        label_names = open(args.label_file).read().split()
        print("[INFO] LABEL FILE %s" % args.label_file)

#    label_names = sg_label_names
    n_classes = len(label_names)
#    n_classes = 3+1
    n_boxes = 5 #3
    chainer.config.train = False

    # initialize CNN model
    print("loading initial model...")
    model = GUINNESS_YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
#    model = MobileYOLOv2(n_classes=n_classes, n_boxes=n_boxes)
    model = YOLOv2Predictor(model)

    if args.pretrained_model is not None:
        if 'snapshot' in args.pretrained_model:
            load_npz(args.pretrained_model, model, path="updater/model:main/")
        else:
#            serializers.load_npz(args.pretrained_model, model) # load temp.npz
            serializers.load_npz(args.pretrained_model, model.predictor) # load model_iter_XXX
    cuda.get_device(args.gpu).use()
    model.to_gpu()

    files = glob.glob(args.data_path + '/Annotations/*.xml')
    ids = []
    for i in range(len(files)):
        id_name = files[i]
        id_name = id_name[:-4]
        id_name = id_name.rsplit('/',1)
        ids.append(id_name[1])

    datasize = len(ids)
#    datasize = 10
    #    print("size=%d" % datasize)
    print("[INFO] #ANNOTATIONS %d" % len(ids))
#    print(ids)

    selected_list = ""

    print("")
    for idx_data in range(datasize):
#    for idx_data in range(0,30):
        id_ = ids[idx_data]
        img_file = os.path.join(data_dir, 'JPEGImages', id_ + '.jpg')
#        print("\033[1A%d/%d" % (idx_data,datasize))

        # --------------------------------------------
        # Detect Objects
        # --------------------------------------------
        # Open Test Image
        orig_img = cv2.imread(str(img_file))
        orig_img = cv2.cvtColor( orig_img, cv2.COLOR_BGR2RGB)
        h, w, ch = orig_img.shape

        img = cv2.resize(orig_img, (args.img_size,args.img_size))
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1).astype(np.float32)[::-1,...]

        # Inference by CNN
        bboxes, labels, scores = model.predict(chainer.cuda.to_gpu(img[None,...]))

        # Save Detection Results
        gts = ''
        for i in range(len(bboxes)):
            ymin, xmax, ymax, xmin = bboxes[i]
            if scores[i] > 0.10:
#                print(labels[i])
#                print(type(labels[i]))
#                print(type(int(labels[i])))
#                print(label_names[int(labels[i])])
#                print(label_names.index(labels[i]))

                line = "%s %f %d %d %d %d\n" % (label_names[int(labels[i])],scores[i],xmin,ymin,xmax,ymax)
                gts += line

        # Draw Detection Results
        result_img = img.transpose(1,2,0).astype(np.uint8)
        for i in range(len(bboxes)):
            ymin, xmax, ymax, xmin = bboxes[i]
#            print("(%d,%d)-(%d,%d)" % (xmin,ymin,xmax,ymax))
#            print("label=%d" % labels[i])
#            print("score=%f" % scores[i])

            if scores[i] > 0.10:
                score_txt = str(scores[i])
                if labels[i] == 0:
                    cv2.rectangle( result_img, (xmin,ymin),(xmax,ymax),(0,255,0), 1)
                    cv2.rectangle( result_img, (xmin,ymin-15),(xmax,ymin),(0,255,0), -1)
                    score_txt = 'car:' + score_txt
                    cv2.putText( result_img, score_txt, (xmin,ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                elif labels[i] == 1:
                    cv2.rectangle( result_img, (xmin,ymin),(xmax,ymax),(0,0,255), 1)
                    cv2.rectangle( result_img, (xmin,ymin-15),(xmax,ymin),(0,0,255), -1)
                    score_txt = 'person:' + score_txt
                    cv2.putText( result_img, score_txt, (xmin,ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                else:
                    cv2.rectangle( result_img, (xmin,ymin),(xmax,ymax),(255,0,0), 1)
                    cv2.rectangle( result_img, (xmin,ymin-15),(xmax,ymin),(255,0,0), -1)
                    score_txt = 'bicycle:' + score_txt
                    cv2.putText( result_img, score_txt, (xmin,ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
#        cv2.imwrite("result.jpg", result_img.astype(np.uint8))

        # --------------------------------------------
        # Draw Ground Truth
        # --------------------------------------------
        '''
        anno = ET.parse(
        os.path.join(data_dir, 'Annotations', id_ + '.xml'))

        for obj in anno.findall('object'):

#            label.append(label_names.index(name))
#            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based

            ymin = int(float(bndbox_anno.find('ymin').text)) - 1
            xmin = int(float(bndbox_anno.find('xmin').text)) - 1
            ymax = int(float(bndbox_anno.find('ymax').text)) - 1
            xmax = int(float(bndbox_anno.find('xmax').text)) - 1

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

            # draw bounding box (for debug)
            cv2.rectangle( result_img, (xmin,ymin),(xmax,ymax),(128,0,128), 3)
        '''

        # fileout detection results
        path = os.path.join(output_dir,id_ + '.txt')
#        print(path)
#        print(gts)
        with open(path, mode='w') as f:
            f.write(gts)               

#        # draw detected bounding box (for debug)
#        cv2.imshow("detect image", result_img)
#        cv2.waitKey(0)


    print("JOB COMPLETE") 
