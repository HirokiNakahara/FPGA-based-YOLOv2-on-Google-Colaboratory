import numpy as np
from chainer.dataset import to_device
from chainer.dataset.convert import _concat_arrays
from chainercv import transforms
from chainercv.links.model.ssd import random_crop_with_bbox_constraints

class Transform(object):

    def __init__(self, n_class, size, random_crop=False,crop_rate=[0.7,1], flip=False, mean=[114.0, 107.2, 98.8],std=[59.6, 57.9, 58.2]):
        # to send cpu, make a copy
        self.size = size
        self.mean = np.array(mean)[...,None,None]
        self.std = np.array(std)[...,None,None]
        self.n_class = n_class
        self.flip = flip
        self.random_crop = random_crop
        self.crop_rate = crop_rate

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 3. Random cropping
        if self.random_crop and np.random.rand() > 0.5:
            next_img, param = random_crop_with_bbox_constraints(
                img, bbox,min_scale=min(self.crop_rate),max_scale=max(self.crop_rate), return_param=True)
            next_bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            if(len(label[param['index']])!=0):
                label = label[param['index']]
                img,bbox=next_img,next_bbox

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = transforms.resize(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        if self.flip:
            img, params = transforms.random_flip(img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(bbox, (self.size, self.size), x_flip=params['x_flip'])

        img -= self.mean
        img /= self.std

        _, height, width = img.shape
        ymin = bbox[:,0]
        xmin = bbox[:,1]
        ymax = bbox[:,2]
        xmax = bbox[:,3]
        one_hot_label = np.eye(self.n_class)[label]
        xs = (xmin + (xmax - xmin)//2) / width
        ws = (xmax - xmin) / width
        ys = (ymin + (ymax - ymin)//2) / height
        hs = (ymax - ymin) / height
        t = [{'label':l,'x':x,'w':w,'y':y,'h':h,'one_hot_label':hot}
                for l,x,w,y,h,hot in zip(label,xs,ws,ys,hs,one_hot_label) ]
        return img, t


class TransformSg(object):

    def __init__(self, n_class, size, random_crop=False,crop_rate=[0.7,1], flip=False, mean=[114.0, 107.2, 98.8],std=[59.6, 57.9, 58.2]):
        # to send cpu, make a copy
        self.size = size
        self.mean = np.array(mean)[...,None,None]
        self.std = np.array(std)[...,None,None]
        self.n_class = n_class
        self.flip = flip
        self.random_crop = random_crop
        self.crop_rate = crop_rate

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 3. Random cropping
        if self.random_crop and np.random.rand() > 0.5:
            next_img, param = random_crop_with_bbox_constraints(
                img, bbox,min_scale=min(self.crop_rate),max_scale=max(self.crop_rate), return_param=True)
            next_bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            if(len(label[param['index']])!=0):
                label = label[param['index']]
                img,bbox=next_img,next_bbox

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = transforms.resize(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        if self.flip:
            img, params = transforms.random_flip(img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(bbox, (self.size, self.size), x_flip=params['x_flip'])
        #from chainercv.visualizations import vis_bbox
        #vis_bbox(img,bbox,label,label_names=voc_bbox_label_names)
        #plt.show()

        img -= self.mean
        img /= self.std

        _, height, width = img.shape
        ymin = bbox[:,0]
        xmin = bbox[:,1]
        ymax = bbox[:,2]
        xmax = bbox[:,3]
        one_hot_label = np.eye(self.n_class)[label]
        xs = (xmin + (xmax - xmin)//2) / width
        ws = (xmax - xmin) / width
        ys = (ymin + (ymax - ymin)//2) / height
        hs = (ymax - ymin) / height
        t = [{'label':l,'x':x,'w':w,'y':y,'h':h,'one_hot_label':hot}
                for l,x,w,y,h,hot in zip(label,xs,ws,ys,hs,one_hot_label) ]
        return img, t


class TransformSgSplit(object):

    def __init__(self, n_class, size, random_crop=False,crop_rate=[0.7,1], flip=False, mean=[114.0, 107.2, 98.8],std=[59.6, 57.9, 58.2]):
        # to send cpu, make a copy
        self.size = size
        self.mean = np.array(mean)[...,None,None]
        self.std = np.array(std)[...,None,None]
        self.n_class = n_class
        self.flip = flip
        self.random_crop = random_crop
        self.crop_rate = crop_rate

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data
        _, height, width = img.shape
        hh,hw=height//2,width//2
        ymin = bbox[0][0]
        xmin = bbox[0][1]
        #ymin = bbox[:,0] #xmin = bbox[:,1]
        #ymax = bbox[:,2] #xmax = bbox[:,3]
        if(ymin<height/2):
            img=img[:,:hh,:]
        else:
            img=img[:,hh:,:]
            bbox[:,0]-=hh
            bbox[:,2]-=hh
        if(xmin<width/2):
            img=img[:,:,:hw]
        else:
            img=img[:,:,hw:]
            bbox[:,1]-=hw
            bbox[:,3]-=hw

        # 3. Random cropping
        if self.random_crop and np.random.rand() > 0.5:
            next_img, param = random_crop_with_bbox_constraints(
                img, bbox,min_scale=min(self.crop_rate),max_scale=max(self.crop_rate), return_param=True)
            next_bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            if(len(label[param['index']])!=0):
                label = label[param['index']]
                img,bbox=next_img,next_bbox

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = transforms.resize(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        if self.flip:
            img, params = transforms.random_flip(img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(bbox, (self.size, self.size), x_flip=params['x_flip'])
        #from chainercv.visualizations import vis_bbox
        #vis_bbox(img,bbox,label,label_names=voc_bbox_label_names)
        #plt.show()

        img -= self.mean
        img /= self.std

        _, height, width = img.shape
        ymin = bbox[:,0]
        xmin = bbox[:,1]
        ymax = bbox[:,2]
        xmax = bbox[:,3]
        one_hot_label = np.eye(self.n_class)[label]
        xs = (xmin + (xmax - xmin)//2) / width
        ws = (xmax - xmin) / width
        ys = (ymin + (ymax - ymin)//2) / height
        hs = (ymax - ymin) / height
        t = [{'label':l,'x':x,'w':w,'y':y,'h':h,'one_hot_label':hot}
                for l,x,w,y,h,hot in zip(label,xs,ws,ys,hs,one_hot_label) ]
        return img, t

def convert_sg(batch, device):
    if len(batch) == 0: raise ValueError('batch is empty')
    result=[
        to_device(device,_concat_arrays([example[0] for example in batch], None)),
        [example[1] for example in batch]]
    return tuple(result)
