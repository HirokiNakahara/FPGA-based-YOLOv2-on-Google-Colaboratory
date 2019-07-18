import numpy as np
import chainer
import os
from chainer import serializers, optimizers, cuda, training
from chainer.training import extension,extensions,updaters
from ..model import YOLOv2Predictor,AlexYOLOv2,MobileYOLOv2,TernaryAlexYOLOv2
#from lib.image_generator import *
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
from datasets import ConcatenatedDataset,CarPersonBboxDataset,VOC3BboxDataset, sg_label_names
from chainer.dataset import to_device
from chainer.dataset.convert import _concat_arrays
from chainercv.visualizations import vis_bbox
from matplotlib import pylab as plt
from chainercv.evaluations import eval_detection_voc
from datasets import ConcatenatedDataset,SgBboxDataset

import chainer.links as L
import chainer.functions as F
from lab import functions as LF
from lab import links as LL
import cv2

# Model Definition{{{
#class AlexYOLOv2Printer(TernaryAlexYOLOv2):
class AlexYOLOv2Printer(AlexYOLOv2):
    def __init__(self, n_classes, n_boxes):
        super().__init__(n_classes, n_boxes)
        self.layer_acts = {}
    def __call__(self, x):
        self.layer_acts['x'] = x
        h = F.leaky_relu(self.bn0(self.conv0(x)), slope=0.125)
        self.layer_acts['conv0_out'] = h.data
        h = F.max_pooling_2d(h, 2)
        self.layer_acts['conv0_p_out'] = h.data
        h = F.leaky_relu(self.bn1(self.conv1(h)), slope=0.125)
        self.layer_acts['conv1_out'] = h.data
        h = F.max_pooling_2d(h, 2)
        self.layer_acts['conv1_p_out'] = h.data
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.125)
        self.layer_acts['conv2_out'] = h.data
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.125)
        self.layer_acts['conv3_out'] = h.data
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.125)
        self.layer_acts['conv4_out'] = h.data
        h = F.max_pooling_2d(h, 2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), slope=0.125)
        h = F.leaky_relu(self.bn6(self.conv6(h)), slope=0.125)
        h = F.leaky_relu(self.bn7(self.conv7(h)), slope=0.125)
        h = self.bias8(self.conv8(h))
        self.layer_acts['layer8']=cuda.to_cpu(h.data)
        return h
#}}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Detection Results')
    parser.add_argument('--img', '-i', type=str, default='test.jpg')
    parser.add_argument('--img_size', '-s', type=int, default=181, help='test image size')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID (negative value uses CPU)')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Initial learning rate for Optimizer')
    args = parser.parse_args()
#    label_names = sg_label_names
#    n_classes = len(label_names)+1
    n_classes = 3+1
    n_boxes = 5 #3
    chainer.config.train = False

    # initialize CNN model
    print("loading initial model...")
    model = AlexYOLOv2Printer(n_classes=n_classes, n_boxes=n_boxes)
    model = YOLOv2Predictor(model)

    if args.pretrained_model is not None:
        if 'snapshot' in args.pretrained_model:
            load_npz(args.pretrained_model, model, path="updater/model:main/")
        else:
            load_npz(args.pretrained_model, model)
    cuda.get_device(args.gpu).use()
    model.to_gpu()


    data_dir = '/home/nakahara/dataset/VOC2012/VOCdevkit/VOC2012/'
    id_list_file = '../dataset_id_list.txt'
    output_dir = './VOC_detection_results/'

    ids = [id_.strip() for id_ in open(id_list_file)]

    # VOC 3 class
    label_names=('car','person','bicycle')
    # VOC original 20 class
    #label_names=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor') # VOC original

    datasize = 10 #len(ids)
    #    print("size=%d" % datasize)

    selected_list = ""

    for idx_data in range(datasize):

        img_file = os.path.join(data_dir, 'JPEGImages', id_ + '.jpg')
        tmp_img = cv2.imread(img_file)
        #    h, w, ch = tmp_img.shape[:3]

#        orig_img = cv2.imread(args.img)
        orig_img = cv2.imread(tmp_img)
        orig_img = cv2.cvtColor( orig_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(orig_img, (args.img_size,args.img_size))
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1).astype(np.float32)[::-1,...]
        bboxes, labels, scores = model.predict(chainer.cuda.to_gpu(img[None,...]))

        gts = ''

        for i in range(len(bboxes)):
            ymin, xmax, ymax, xmin = bboxes[i]
            if scores[i] > 0.10:
                print(label_names.index(labels[i]))

                line = "%s %d %d %d %d\n" % (label_names.index(labels[i]),xmin,ymin,xmax,ymax)
                gts += line

        '''
        result_img = img.transpose(1,2,0).astype(np.uint8)
        for i in range(len(bboxes)):
            ymin, xmax, ymax, xmin = bboxes[i]
            print("(%d,%d)-(%d,%d)" % (xmin,ymin,xmax,ymax))
            print("label=%d" % labels[i])
            print("score=%f" % scores[i])

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
        cv2.imwrite("result.jpg", result_img.astype(np.uint8))
        '''

        path = os.path.join(output_dir,id_ + '.txt')
        print(path)
        print(gts)
        with open(path, mode='w') as f:
            f.write(gts)                
