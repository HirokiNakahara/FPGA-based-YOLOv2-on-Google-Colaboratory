#!/bin/sh
python gen_ground_truth.py -d $3 -o mAP/VOC_ground_truth -s $1 -l voc3_label.txt
python gen_detection_results.py -s $1 -g 0 --pretrained_model $2 -l voc3 -d $3 -o mAP/VOC_detection_results
python mAP/main.py -na -np -gt VOC_ground_truth -dr VOC_detection_results

#python gen_ground_truth.py -d /home/nakahara/dataset/VOC2007Test/VOCdevkit/VOC2007 -o mAP/VOC_ground_truth -s 213 -l voc3_label.txt
#python gen_detection_results.py -s 213 -g 0 --pretrained_model dense_BinAlexYOLOv2_1/model_epoch_2000 -l voc3 -d /home/nakahara/dataset/VOC2007Test/VOCdevkit/VOC2007 -o mAP/VOC_detection_results
#python mAP/main.py -na -np -gt VOC_ground_truth -dr VOC_detection_results

#python gen_ground_truth.py -d /home/nakahara/SparseYOLOv2_GUINNESS_template0.9/gpu_train/datasets/selected_VOC07+12_r2 -o mAP/VOC_ground_truth -s 213 -l voc3_label.txt
#python gen_detection_results.py -s 213 -g 0 --pretrained_model dense_BinAlexYOLOv2_1/model_epoch_2000 -l voc3 -d /home/nakahara/SparseYOLOv2_GUINNESS_template0.9/gpu_train/datasets/selected_VOC07+12_r2/ -o mAP/VOC_detection_results
#python mAP/main.py -na -np -gt VOC_ground_truth -dr VOC_detection_results
