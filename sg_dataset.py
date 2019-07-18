import numpy as np
import os
import warnings

import chainer

from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image

sg_label_names=('logo', 'logo_A', 'logo_I', 'logo_C', 'logo_e',
            'clk1', 'clk2', 'clk3', 'clk4', 'clk5',
            'ban1', 'ban2', 'ban3', 'ban4', 'ban5',
            'ban6', 'ban7', 'ban8', 'ban9', 'ban10')

class SgBboxDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir='auto'):
        if data_dir == 'auto':
            data_dir = os.path.join(os.path.expanduser("~"),'.chainer/dataset/pfnet/chainercv/sg')

        #id_list_file = os.path.join( data_dir, 'classifier_sdc.txt')
        #id_list_file = os.path.join( data_dir, 'banner.txt')
        id_list_file = os.path.join( data_dir, 'banner_split.txt')

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.class_names= sg_label_names

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
        id_ = self.ids[i]
        # Load a image
        img_file = os.path.join(self.data_dir, id_ + '.jpg')
        img = read_image(img_file, color=True)
        input_h,input_w=img.shape[1:]

        # Load a annotation
        anno = np.loadtxt(os.path.join(self.data_dir, id_ + '.txt'))
        #if anno.size!=5:
            #raise ValueError('num of class per image is expected to be one')

        if anno.size==5:
            anno=anno[None,...]
        x,y,w,h = anno[:,1],anno[:,2],anno[:,3],anno[:,4]
        xmin=(x-w/2)*input_w
        xmax=(x+w/2)*input_w
        ymin=(y-h/2)*input_h
        ymax=(y+h/2)*input_h

        bbox = np.round(np.array([ymin, xmin, ymax, xmax]).T).astype(np.float32)
        label = np.stack(anno[:,0]).astype(np.int32)
        return img, bbox, label

