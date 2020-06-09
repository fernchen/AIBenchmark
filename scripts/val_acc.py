import sys
from collections import namedtuple
import time
import os
os.environ['GLOG_minloglevel'] = '2'
import yaml

import cv2
import caffe
import numpy as np

from utils import ImageProcess
from utils import alexnet_preprocess, resnet_preprocess
from utils import mobilenet_preprocess, inception_preprocess
from utils import squeezenet_preprocess, densenet_preprocess
from utils import vgg_preprocess


with open('./configs.yaml', 'r') as f:
    cfg_info = yaml.load(f)


def main(current_model):
    if 'name' not in current_model:
        current_name = "model"
    else:
        current_name = current_model['name']

    caffe.set_mode_gpu()

    img2label = dict()
    val_txt = cfg_info['val_txt']
    if 'val_txt' in current_model:
        val_txt = current_model['val_txt']
    with open(val_txt, 'r') as f:
        for line in f.readlines():
            img_path, label = line.strip().split(" ")
            img2label[cfg_info['val_imgs_dir']+img_path] = int(label)

    net = caffe.Net(current_model['proto_file'], current_model['blob_file'], caffe.TEST)

    cnt = 0
    top1 = 0
    top5 = 0
    t_total_start = time.time()
    t_mid_start = t_total_start
    for img_path, label in img2label.items():
        net.blobs['data'].data[...] = eval(current_model['pre_func_name'])(img_path, **(current_model['params']))
        out = net.forward()
        for _, v in out.items():
            prob = np.squeeze(v)
            idx = np.argsort(prob)
            idx = idx[::-1]
            if idx[0] == label:
                top1 += 1
            if label in idx[:5]:
                top5 += 1
        cnt += 1
        if cnt % 100 == 0:
            t_mid_end = time.time()
            print("{}_num-{} top1:{}, top5:{}, time cost: {}s".format(
                current_name, cnt, (top1 / cnt), (top5 /
                                                  cnt), (t_mid_end - t_mid_start)
            ))
            t_mid_start = t_mid_end
    t_total_end = time.time()
    print("Total time cost: {}s".format(t_total_end - t_total_start))
    print("{} total acc: top1: {}, top5: {}".format(
        current_name, top1/cnt, top5/cnt))


if __name__ == "__main__":
    current_model = cfg_info['Vgg16']
    main(current_model)
