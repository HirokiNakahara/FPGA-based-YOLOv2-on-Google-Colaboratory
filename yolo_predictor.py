import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, initializers, reporter
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
from lib.functions import *
from lib.image_generator import *

class YOLOv2Predictor(Chain):
    def __init__(self, predictor, conf_scale=0.1, unstable_seen=0):
        super(YOLOv2Predictor, self).__init__(predictor=predictor)
        # from car person
        #self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]
        # from chainercv
        #self.anchors = [[1.73145, 1.3221], [4.00944, 3.19275], [8.09892, 5.05587], [4.84053, 9.47112], [10.0071, 11.2364]]
        #self.anchors = [[0.79913077, 0.6102], [1.85051077, 1.47357692], [3.73796308, 2.33347846], [2.23409077, 4.37128615], [4.61866154, 5.18603077]]
        #self.anchors = [[0.79913077, 0.6102]]
#        self.anchors = [[0.79913077, 0.6102], [1.85051077, 1.47357692], [3.73796308, 2.33347846], [2.23409077, 4.37128615], [4.61866154, 5.18603077]]
#        self.anchors = [[0.8, 0.6],[0.8, 0.6]]
#        self.anchors = [[0.8, 0.6], [1.9, 1.5], [3.7, 2.3]]

        # from chainercv
#        self.anchors = [[1.73145, 1.3221], [4.00944, 3.19275], [8.09892, 5.05587], [4.84053, 9.47112], [10.0071, 11.2364]]
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]

        self.thresh = 0.6 # keep conf value threshold
        self.seen = 0
#        self.unstable_seen = 8000 #49000 #12000 #3000 #5000 # 0 for resume training #5000 for initial training
        # 24 batch x 3000 epochs = 72000

        self.unstable_seen = unstable_seen

        self.conf_scale = conf_scale
    def __call__(self, input_x, t):
        output = self.predictor(input_x)

        batch_size, _, grid_h, grid_w = output.shape
        self.seen += batch_size

        x, y, w, h, conf, prob = F.split_axis(F.reshape(output, (batch_size, self.predictor.n_boxes, self.predictor.n_classes+5, grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
        x = F.sigmoid(x) # xのactivation
        y = F.sigmoid(y) # yのactivation
        conf = F.sigmoid(conf) # confのactivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob) # probablitiyのacitivation

        # 教師データの用意
        tw = np.zeros(w.shape, dtype=np.float32) # wとhが0になるように学習(e^wとe^hは1に近づく -> 担当するbboxの倍率1)
        th = np.zeros(h.shape, dtype=np.float32)
        tx = np.tile(0.5, x.shape).astype(np.float32) # 活性化後のxとyが0.5になるように学習()
        ty = np.tile(0.5, y.shape).astype(np.float32)

        if self.seen < self.unstable_seen: # centerの存在しないbbox誤差学習スケールは基本0.1
#            box_learning_scale = np.tile(0.1, x.shape).astype(np.float32)
            box_learning_scale = np.tile(self.conf_scale, x.shape).astype(np.float32)
        else:
            box_learning_scale = np.tile(0, x.shape).astype(np.float32)

#        print((self.seen,self.unstable_seen,self.conf_scale))

        tconf = np.zeros(conf.shape, dtype=np.float32) # confidenceのtruthは基本0、iouがthresh以上のものは学習しない、ただしobjectの存在するgridのbest_boxのみ真のIOUに近づかせる
        conf_learning_scale = np.tile(0.1, conf.shape).astype(np.float32)

        tprob = prob.data.copy() # best_anchor以外は学習させない(自身との二乗和誤差 = 0)

        # 全bboxとtruthのiouを計算(batch単位で計算する)
        x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape[1:]))
        y_shift = Variable(np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape[1:]))
        w_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)), w.shape[1:]))
        h_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)), h.shape[1:]))
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        best_ious = []
        for batch in range(batch_size):
            n_truth_boxes = len(t[batch])
            if n_truth_boxes<1: continue
            box_x = (x[batch] + x_shift) / grid_w
            box_y = (y[batch] + y_shift) / grid_h
            box_w = F.exp(w[batch]) * w_anchor / grid_w
            box_h = F.exp(h[batch]) * h_anchor / grid_h

            ious = []
            for truth_index in range(n_truth_boxes):
                truth_box_x = Variable(np.broadcast_to(np.array(t[batch][truth_index]["x"], dtype=np.float32), box_x.shape))
                truth_box_y = Variable(np.broadcast_to(np.array(t[batch][truth_index]["y"], dtype=np.float32), box_y.shape))
                truth_box_w = Variable(np.broadcast_to(np.array(t[batch][truth_index]["w"], dtype=np.float32), box_w.shape))
                truth_box_h = Variable(np.broadcast_to(np.array(t[batch][truth_index]["h"], dtype=np.float32), box_h.shape))
                truth_box_x.to_gpu(), truth_box_y.to_gpu(), truth_box_w.to_gpu(), truth_box_h.to_gpu()
                ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h), Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)).data.get())  
            ious = np.array(ious)
            best_ious.append(np.max(ious, axis=0))
        best_ious = np.array(best_ious)

        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする(truthの周りのgridはconfをそのまま維持)。
        tconf[best_ious > self.thresh] = conf.data.get()[best_ious > self.thresh]
        conf_learning_scale[best_ious > self.thresh] = 0

        # objectの存在するanchor boxのみ、x、y、w、h、conf、probを個別修正
        abs_anchors = self.anchors / np.array([grid_w, grid_h])
        for batch in range(batch_size):
            n_truth_boxes = len(t[batch])
            if n_truth_boxes<1: continue
            for truth_box in t[batch]:
                truth_w = int(float(truth_box["x"]) * grid_w)
                truth_h = int(float(truth_box["y"]) * grid_h)
                truth_n = 0
                best_iou = 0.0
                for anchor_index, abs_anchor in enumerate(abs_anchors):
                    iou = box_iou(Box(0, 0, float(truth_box["w"]), float(truth_box["h"])), Box(0, 0, abs_anchor[0], abs_anchor[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_n = anchor_index

                # objectの存在するanchorについて、centerを0.5ではなく、真の座標に近づかせる。anchorのスケールを1ではなく真のスケールに近づかせる。学習スケールを1にする。
                #print(box_learning_scale.shape,batch,truth_n,truth_h,truth_w)
                box_learning_scale[batch, truth_n, :, truth_h, truth_w] = 1.0 
                tx[batch, truth_n, :, truth_h, truth_w] = float(truth_box["x"]) * grid_w - truth_w 
                ty[batch, truth_n, :, truth_h, truth_w] = float(truth_box["y"]) * grid_h - truth_h
                tw[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box["w"]) / abs_anchors[truth_n][0])
                th[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box["h"]) / abs_anchors[truth_n][1])
                tprob[batch, :, truth_n, truth_h, truth_w] = 0
                tprob[batch, int(truth_box["label"]), truth_n, truth_h, truth_w] = 1

                # IOUの観測
                full_truth_box = Box(float(truth_box["x"]), float(truth_box["y"]), float(truth_box["w"]), float(truth_box["h"]))
                predicted_box = Box(
                    (x[batch][truth_n][0][truth_h][truth_w].data.get() + truth_w) / grid_w, 
                    (y[batch][truth_n][0][truth_h][truth_w].data.get() + truth_h) / grid_h,
                    np.exp(w[batch][truth_n][0][truth_h][truth_w].data.get()) * abs_anchors[truth_n][0],
                    np.exp(h[batch][truth_n][0][truth_h][truth_w].data.get()) * abs_anchors[truth_n][1]
                )
                predicted_iou = box_iou(full_truth_box, predicted_box)
                tconf[batch, truth_n, :, truth_h, truth_w] = predicted_iou
                conf_learning_scale[batch, truth_n, :, truth_h, truth_w] = 10.0

            # debug prints
            maps = F.transpose(prob[batch], (2, 3, 1, 0)).data

        # loss計算
        tx, ty, tw, th, tconf, tprob = Variable(tx), Variable(ty), Variable(tw), Variable(th), Variable(tconf), Variable(tprob)
        box_learning_scale, conf_learning_scale = Variable(box_learning_scale), Variable(conf_learning_scale)
        tx.to_gpu(), ty.to_gpu(), tw.to_gpu(), th.to_gpu(), tconf.to_gpu(), tprob.to_gpu()
        box_learning_scale.to_gpu()
        conf_learning_scale.to_gpu()

        x_loss = F.sum((tx - x) ** 2 * box_learning_scale) / 2
        y_loss = F.sum((ty - y) ** 2 * box_learning_scale) / 2
        w_loss = F.sum((tw - w) ** 2 * box_learning_scale) / 2
        h_loss = F.sum((th - h) ** 2 * box_learning_scale) / 2
        c_loss = F.sum((tconf - conf) ** 2 * conf_learning_scale) / 2
        p_loss = F.sum((tprob - prob) ** 2) / 2

        #print("x_loss: %f  y_loss: %f  w_loss: %f  h_loss: %f  c_loss: %f   p_loss: %f" % 
        #    (F.sum(x_loss).data, F.sum(y_loss).data, F.sum(w_loss).data, F.sum(h_loss).data, F.sum(c_loss).data, F.sum(p_loss).data)
        #)

#        x_loss = F.sum( pow((tx - x) * box_learning_scale,2))
#        y_loss = F.sum( pow((ty - y) * box_learning_scale,2))
#        w_loss = F.sum( pow((tw - w) * box_learning_scale,2))
#        h_loss = F.sum( pow((th - h) * box_learning_scale,2))
#        c_loss = F.sum( pow((tconf - conf) * conf_learning_scale,2))
#        p_loss = F.sum( pow((tprob - prob),2))


        loss = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss
        reporter.report({'x_loss': x_loss}, self)
        reporter.report({'y_loss': y_loss}, self)
        reporter.report({'w_loss': w_loss}, self)
        reporter.report({'h_loss': h_loss}, self)
        reporter.report({'c_loss': c_loss}, self)
        reporter.report({'p_loss': p_loss}, self)
        reporter.report({'loss': loss}, self)

        return loss

    def init_anchor(self, anchors):
        self.anchors = anchors

    def s16(value):
        return -(value & 0b1000000000000000) | (value & 0b0111111111111111)

    def predict(self, input_x):#, output_):
        #output = output_.reshape(1,40,6,6).astype(np.float32)
        detection_thresh = 0.1 #0.1
        iou_thresh = 0.5 #0.35
        output = self.predictor(input_x)

        batch_size, input_channel, input_h, input_w = input_x.shape
        batch_size, _, grid_h, grid_w = output.shape
        n_boxes=self.predictor.n_boxes
        n_classes=self.predictor.n_classes
#        print("n_classes",n_classes)

        tmp=F.reshape(
                output,
                (batch_size, n_boxes, n_classes+5, grid_h, grid_w))
        x, y, w, h, conf, prob = F.split_axis(tmp, (1, 2, 3, 4, 5), axis=2)

        def prnt(d,name):
            data=np.squeeze(d)
            print(name)
            print(data.shape)
            for i,p in zip(range(data.size),data.reshape(-1)):
                print(p,end='')
                if((i+1)%5==0):print('')
                if((i+1)%25==0):print('')

            #print("%s %.8f %.8f %.8f" % (name,d.reshape(-1)[0],d.reshape(-1)[32],d.reshape(-1)[64]))
#        prnt(x.data,'x')
#        prnt(y.data,'y')
#        prnt(w.data,'w')
#        prnt(h.data,'h')
#        prnt(conf.data,'conf')
#        prnt(prob.data,'prob')

        x = F.sigmoid(x) # xのactivation
        y = F.sigmoid(y) # yのactivation
        conf = F.sigmoid(conf) # confのactivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob) # probablitiyのacitivation
        prob = F.transpose(prob, (0, 2, 1, 3, 4))

        x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape))

        y_shift = Variable(np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape))
        w_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (n_boxes, 1, 1, 1)), w.shape))
        h_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (n_boxes, 1, 1, 1)), h.shape))

        x,y=cuda.to_cpu(x.data),cuda.to_cpu(y.data)
        w,h=cuda.to_cpu(w.data),cuda.to_cpu(h.data)
        conf,prob = cuda.to_cpu(conf.data), cuda.to_cpu(prob.data)

        box_x = (x + x_shift.data) / grid_w
        box_y = (y + y_shift.data) / grid_h
        box_w = np.exp(w) * w_anchor.data / grid_w
        box_h = np.exp(h) * h_anchor.data / grid_h

#        return box_x, box_y, box_w, box_h, Variable(cuda.to_cpu(conf.data)), Variable(cuda.to_cpu(prob.data))
#        x, y, w, h, conf, prob = pred
        x, y, w, h = box_x, box_y, box_w, box_h
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (n_boxes, grid_h, grid_w)).data

        conf = F.reshape(conf, (n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (n_boxes, n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > detection_thresh
        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "label": prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax(),
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf" : conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(x[detected_indices][i]*input_w,
                            y[detected_indices][i]*input_h,
                            w[detected_indices][i]*input_w,
                            h[detected_indices][i]*input_h).crop_region(input_h, input_w)
            })
        nms_results = nms(results, iou_thresh)
        bboxes=[]
        labels=[]
        scores=[]
        for result in nms_results:
#            print(result["probs"].max()*result["conf"]*100)
            if result["probs"].max()*result["conf"]*100 >= 20.0:
#            if result["probs"].max()*result["conf"]*100 >= 0.0:
                x1, y1 = result["box"].int_left_top()
                x2, y2 = result["box"].int_right_bottom()
                bboxes.append([y1,x2,y2,x1])
                scores.append(result["probs"].max()*result["conf"])
                labels.append(result["probs"].argmax())
#                print("prob,", result["probs"].max())
#                print("conf,", result["conf"])
#                print("score,", result["probs"].max()*result["conf"])
#                print(bboxes)
        return bboxes, labels, scores

####
####        xmin=(box_x*2-box_w)*input_w//2
####        xmax=(box_x+box_w/2)*input_w
####        ymin=(box_y*2-box_h)*input_h//2
####        ymax=(box_y+box_h/2)*input_h
####        print(xmin.shape,prob.data.shape,type(prob.data))
####
####        #return box_x, box_y, box_w, box_h, Variable(cuda.to_cpu(conf.data)), Variable(cuda.to_cpu(prob.data))
####        return bboxes, labels, scores

#    ###### rewrite into cpp script ######
#    def predict(self, input_x, output_):
#        output = output_.reshape(1,40,6,6).astype(np.float32)
#
#        batch_size, input_channel, input_h, input_w = input_x.shape
#        batch_size, _, grid_h, grid_w = output.shape
#        tmp=F.reshape(
#                output,
#                (batch_size, self.predictor.n_boxes, self.predictor.n_classes+5, grid_h, grid_w))
#        x, y, w, h, conf, prob = F.split_axis(tmp, (1, 2, 3, 4, 5), axis=2)
#
#        x = F.sigmoid(x) # xのactivation
#        y = F.sigmoid(y) # yのactivation
#        conf = F.sigmoid(conf) # confのactivation
#        prob = F.transpose(prob, (0, 2, 1, 3, 4))
#        prob = F.softmax(prob) # probablitiyのacitivation
#        prob = F.transpose(prob, (0, 2, 1, 3, 4))
#
#        x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape))
#
#        y_shift = Variable(np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape))
#        w_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)), w.shape))
#        h_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)), h.shape))
#
#        box_x = (x + x_shift) / grid_w
#        box_y = (y + y_shift) / grid_h
#        box_w = F.exp(w) * w_anchor / grid_w
#        box_h = F.exp(h) * h_anchor / grid_h
#
##        return box_x, box_y, box_w, box_h, conf, prob
#        return box_x, box_y, box_w, box_h, Variable(cuda.to_cpu(conf.data)), Variable(cuda.to_cpu(prob.data))
#
