# http://tosaka2.hatenablog.com/entry/2017/11/17/194051
# out mapチャネルごと しきい値を決めるように変更
# iterative pruningの追加
import chainer
import chainer.links as L
from chainer import training,cuda

import numpy as np

def create_layer_mask(weights, pruning_rate, is_mapwise=True):
    # W shape (n_in_map,n_out_map,kernel_H,kernel_W)
    if weights.data is None:
        raise Exception("Some weights of layer is None.")

    abs_W = np.abs(cuda.to_cpu(weights.data))
    if is_mapwise:
        data = np.sort(abs_W.reshape(abs_W.shape[0],-1))
        num_prune = int(data.shape[1] * pruning_rate)
        idx_prune = min(num_prune, data.shape[1]-1)
        threshould = data[:,idx_prune]
    else:
        data = np.sort(abs_W.reshape(-1))
        num_prune = int(len(data) * pruning_rate)
        idx_prune = min(num_prune, len(data)-1)
        threshould = data[idx_prune]

    if is_mapwise:
        mask=(abs_W >= threshould[...,None,None,None]).astype(np.float32)
    else:
        mask=(abs_W >= threshould).astype(np.float32)
    return cuda.to_gpu(mask)

'''Returns a trainer extension to fix pruned weight of the model.
'''
def create_model_mask(model, pruning_rate):
    masks = {}
    for name, link in model.namedlinks():
        # specify pruned layer
        if type(link) not in (L.Convolution2D, L.Linear):
            continue
        print(name,'sparse mask is created')
        mask = create_layer_mask(link.W, pruning_rate)
        masks[name] = mask
    return masks


def prune_weight(model, masks):
    for name, link in model.namedlinks():
        if name not in masks.keys():
            continue
        mask = masks[name]
        link.W.data = link.W.data * mask

'''Returns a trainer extension to fix pruned weight of the model.
'''
def pruned(model, masks):
    @training.make_extension(trigger=(1, 'iteration'))
    def _pruned(trainer):
        prune_weight(model, masks)
    return _pruned

''' iteratice pruning
    model: cnn model
    st_rate:  first pruning rate
    end_rate: pruning rate to reach at final epoch
    trigger_epoch: update the pruning mask each this number of epoch
'''
class Iter_create_mask:
    def __init__(self, model, n_epoch, st_rate=0.4, end_rate=0.95, trigger_epoch=20):
        self.model=model
        self.trigger_epoch=trigger_epoch
        self.interval=(end_rate-st_rate)/((n_epoch-trigger_epoch)/trigger_epoch)
        self.pr_rate=st_rate
        self.end_rate=end_rate
        self.masks=create_model_mask(self.model, st_rate)
        prune_weight(self.model, self.masks)

    def update_mask(self):
        @training.make_extension(trigger=(self.trigger_epoch,'epoch'))
        def _update_mask(trainer):
            self.pr_rate+=self.interval
            self.masks=create_model_mask(self.model, min(self.pr_rate, self.end_rate))
            print('update pruning rate', min(self.pr_rate, self.end_rate))
        return _update_mask

    def pruned(self):
        @training.make_extension(trigger=(1, 'iteration'))
        def _pruned(trainer):
            prune_weight(self.model, self.masks)
        return _pruned

