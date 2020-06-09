import sys

import cv2
import numpy as np


class ModelType(object):
    Resnet = 0
    Mobilenet = 1
    Alexnet = 2
    Inception = 3
    NoModel = -1


def str2ModelType(name):
    res = None
    if 'resnet' in name.lower():
        res = ModelType.Resnet
    elif 'mobilenet' in name.lower():
        res = ModelType.Mobilenet
    elif 'alexnet' in name.lower():
        res = ModelType.Alexnet
    elif 'inception' in name.lower():
        res = ModelType.Inception
    else:
        res = ModelType.NoModel
    return res


class ImageProcess(object):
    def __init__(self, img_path=None):
        self.img_path = img_path
        self.data = None

    def load_image(self, img_path=None):
        if img_path:
            self.img_path = img_path
        try:
            self.data = cv2.imread(self.img_path)
        except IOError:
            sys.exit("Fail to load the image:{}".format(self.img_path))
        self.data = self.data.astype(np.float32)
        # bad imgs: gray images and rgba images.
        if len(self.data.shape) == 2:
            self.data = self.data[:, :, np.newaxis]
            self.data = np.tile(self.data, (1, 1, 3))
        elif self.data.shape[2] == 4:
            self.data = self.data[:, :, :3]

    def resize_image(self, short_side_len=256):
        if self.data is None:
            sys.exit("Please load image first.")
        if isinstance(short_side_len, (list, tuple)):
            self.data = cv2.resize(self.data, tuple(short_side_len),
                                   interpolation=cv2.INTER_CUBIC)
        else:
            height, width, _ = self.data.shape
            new_height = height * short_side_len // min(self.data.shape[:2])
            new_width = width * short_side_len // min(self.data.shape[:2])
            self.data = cv2.resize(self.data, (new_width, new_height),
                                   interpolation=cv2.INTER_CUBIC)

    def crop_image(self, crop_len=224):
        if self.data is None:
            sys.exit("Please load image first.")
        if crop_len <= 0:
            sys.exit("Please input correct crop_len which should be "
                     "no less than 0.")
        height, width, _ = self.data.shape
        startx = width // 2 - (crop_len//2)
        starty = height // 2 - (crop_len//2)
        self.data = self.data[starty:starty+crop_len, startx:startx+crop_len]

    def transpose(self, trans_order=None):
        if self.data is None:
            sys.exit("Please load image first.")
        if trans_order is not None:
            self.data = np.transpose(self.data, trans_order)

    def sub_mean_val(self, mean_val=None):
        if self.data is None:
            sys.exit("Please load image first.")
        if mean_val is None or (not isinstance(mean_val, np.ndarray)):
            sys.exit("Please load mean_val with type np.ndarray.")
        if mean_val.ndim == 3:  # pixel-wise mean
            self.data -= mean_val
        elif mean_val.ndim == 1:
            if len(mean_val) == 1:
                self.data -= mean_val
            elif len(mean_val) == 3:
                self.data[0, :, :] -= mean_val[0]
                self.data[1, :, :] -= mean_val[1]
                self.data[2, :, :] -= mean_val[2]
            else:
                sys.exit("Incorrect mean_val:{}".format(mean_val))

    def input_scale(self, scale=None):
        if self.data is None:
            sys.exit("Please load image first.")
        if scale is not None:
            self.data *= scale


def convert_binaryproto2npy(b_path):
    try:
        import caffe
    except ImportError:
        sys.exit("Please install caffe")
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(b_path, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_path = b_path.replace('.binaryproto', '.npy')
    np.save(npy_path, arr)


def get_mean_val(mean_val):
    if isinstance(mean_val, str):
        out = np.squeeze(np.load(mean_val))
    elif isinstance(mean_val, (list, tuple)):
        out = np.array(mean_val, dtype=np.float32)
    else:
        out = np.array([0, 0, 0], dtype=np.float32)
    return out


def alexnet_preprocess(img_path, mean_val=None, resize_size=None,
                        crop_size=None):
    mean_val = get_mean_val(mean_val)
    mean_val = np.transpose(mean_val, (1, 2, 0))
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.sub_mean_val(mean_val=mean_val)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))

    return img.data


def resnet_preprocess(img_path, mean_val=None, resize_size=None,
                        crop_size=None):
    mean_val = get_mean_val(mean_val)
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))
    img.sub_mean_val(mean_val=mean_val)

    return img.data


def mobilenet_preprocess(img_path, mean_val=None, input_scale=None,
                            resize_size=None, crop_size=None):
    mean_val = get_mean_val(mean_val)
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))
    img.sub_mean_val(mean_val=mean_val)
    img.input_scale(scale=input_scale)

    return img.data


def inception_preprocess(img_path, mean_val=None, resize_size=None,
                            crop_size=None, input_scale=None):
    mean_val = get_mean_val(mean_val)
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))
    img.sub_mean_val(mean_val=mean_val)

    if input_scale is not None:
        img.input_scale(scale=input_scale)

    return img.data


def squeezenet_preprocess(img_path, mean_val=None, resize_size=None,
                        crop_size=None):
    mean_val = get_mean_val(mean_val)
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))
    img.sub_mean_val(mean_val=mean_val)

    return img.data


def densenet_preprocess(img_path, mean_val=None, input_scale=None,
                            resize_size=None, crop_size=None):
    mean_val = get_mean_val(mean_val)
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))
    img.sub_mean_val(mean_val=mean_val)
    img.input_scale(scale=input_scale)

    return img.data


def vgg_preprocess(img_path, mean_val=None, resize_size=None,
                        crop_size=None):
    mean_val = get_mean_val(mean_val)
    img = ImageProcess(img_path)
    img.load_image()
    img.resize_image(short_side_len=resize_size)
    img.crop_image(crop_len=crop_size)
    img.transpose(trans_order=(2, 0, 1))
    img.sub_mean_val(mean_val=mean_val)

    return img.data


def prepare_dataset_label():
    img_name_list = r'/home/fern/code/projects/AIBenchmark/data/ILSVRC2012/val.txt'
    label_list = r'/home/fern/下载/ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt'

    with open(img_name_list, 'r') as f:
        img_names = f.readlines()
    with open(label_list, 'r') as f:
        labels = f.readlines()
    res = list()
    for idx, img_name in enumerate(img_names):
        label = labels[idx].strip()
        img = img_name.split(' ')[0].strip()
        res.append(img+' '+label+'\n')
    with open('/home/fern/code/projects/AIBenchmark/data/ILSVRC2012/val_2014.txt', 'w') as f:
        f.writelines(res)


if __name__ == "__main__":
    # b_path = '/home/fern/code/projects/AIBenchmark/models/caffe/resnet/ResNet_mean.binaryproto'
    # convert_binaryproto2npy(b_path)
    prepare_dataset_label()
