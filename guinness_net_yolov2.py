# -----------------------------------------------------------------------
# template_yolov2_cnn.py
# Template File for a YOLOv2 CNN Generator
#
# Creation Date   : 01/Nov./2018
# Copyright (C) <2017> Hiroki Nakahara, All rights reserved.
# 
# Released under the Tokyo Institute of Technology License.
# -----------------------------------------------------------------------

import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, initializers
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
#from lib.utils import *
#from lib.functions import *
#from guinnesslib import links as GL

import math
import six
#import bst

'''
import link_binary_linear as BL
import bst
import link_binary_conv2d as BC
import link_integer_conv2d as IC
import link_sparse_conv2d as SC
import link_ternary_conv2d as TC

import sys
sys.path.append('./')
from function_binary_conv2d import func_convolution_2d
from function_integer_conv2d import func_convolution_2d
from function_sparse_conv2d import func_convolution_2d
from function_ternary_conv2d import func_convolution_2d
'''

from subprocess import check_call
import subprocess

import socket

class GUINNESS_YOLOv2(Chain):
    def __init__(self, n_classes, n_boxes):
        initializer = initializers.Normal(scale=0.000500, dtype=None) # 0.0005
        super(GUINNESS_YOLOv2, self).__init__()
        with self.init_scope():
            self.conv0=L.Convolution2D(3,64,ksize=11, stride=4, pad=1, initialW=initializer)
            self.bn0=L.BatchNormalization(64, use_beta=False)
            self.conv1=L.Convolution2D(64,64,ksize=5, stride=1, pad=1, initialW=initializer)
            self.bn1=L.BatchNormalization(64, use_beta=False)
            self.conv2=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
            self.bn2=L.BatchNormalization(64, use_beta=False)
            self.conv3=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
            self.bn3=L.BatchNormalization(64, use_beta=False)
            self.conv4=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
            self.bn4=L.BatchNormalization(64, use_beta=False)
            self.conv5=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
            self.bn5=L.BatchNormalization(64, use_beta=False)
#            self.conv6=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
#            self.bn6=L.BatchNormalization(64, use_beta=False)
#            self.conv7=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
#            self.bn7=L.BatchNormalization(64, use_beta=False)
#            self.conv8=L.Convolution2D(64,64,ksize=3, stride=1, pad=1, initialW=initializer)
#            self.bn8=L.BatchNormalization(64, use_beta=False)
            self.conv6=L.Convolution2D(64,n_boxes * (5 + n_classes),ksize=1, stride=1, pad=0, initialW=initializer)

        self.train = False
        self.finetune = False
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def __call__(self, x):
#        batch_size = x.data.shape[0]
        h = F.leaky_relu(self.bn0(self.conv0(x)), slope=0.125)
        h = F.max_pooling_2d(h, 2)
        h = F.leaky_relu(self.bn1(self.conv1(h)), slope=0.125)
        h = F.max_pooling_2d(h, 2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.125)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.125)
        h = F.max_pooling_2d(h, 2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.125)
        h = F.leaky_relu(self.bn5(self.conv5(h)), slope=0.125)
#        h = F.leaky_relu(self.bn6(self.conv6(h)), slope=0.125)
#        h = F.leaky_relu(self.bn7(self.conv7(h)), slope=0.125)
#        h = F.leaky_relu(self.bn8(self.conv8(h)), slope=0.125)
        h = self.conv6(h)
        return h

