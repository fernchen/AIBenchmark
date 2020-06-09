## AIBenchmark

This is a collection of common basic AI models, mainly collected from official repos and third-party implementation. This repo mainly includes the following contents: paper links, original models, measured performance data and test scripts.

> All the model data can be found at:
>
> - [baiduyun](https://pan.baidu.com/s/1C1fjHqpYG3cqtZ1MqBxWSQ), code: `5nd3`
> - [google driver](https://drive.google.com/drive/folders/1r4Q5cDgP7JwJlMG60ODO3tzYPTuxa4Y5?usp=sharing)
>
> Host computer infomation:
>
> - Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
> - GeForce RTX 2070 super

### 1 Classification Models
ILSVRC2012 dataset is used to verify and test the classification models, which can be downloaded at [Imagenet](http://www.image-net.org/) and [Imagenet Torrents](https://academictorrents.com/collection/imagenet-2012).

#### 1.1 Alexnet

> Repo link：[caffe-alexnet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

##### 1.1.1 paper

Alexnet:  [ImageNet Classification with Deep Convolutional Neural Networks](reference/Alexnet:imagenet-classification-with-deep-convolutional-neural-networks.pdf)

##### 1.1.2 preprocess

1-crop method is adopted，resize to (256, 256)，central crop: 227， mean value file: [imagenet-mean](data/ILSVRC2012/imagenet_mean.npy)

```Python
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
```

##### 1.1.3 test results

| model name | top1    | top5    |
| ---------- | ------- | ------- |
| Alexnet    | 0.56184 | 0.79474 |



#### 1.2 Mobilenet

> Repo link： [shicai repo](https://github.com/shicai/MobileNet-Caffe)[caffe]

##### 1.2.1 paper

- Mobilentv1: [MobileNets Efficient-Convolutional-Neural-Networks-for-Mobile-Vision-Applications.pdf](reference/mobilenetv1-MobileNets-Efficient-Convolutional-Neural-Networks-for-Mobile-Vision-Applications.pdf)

- Mobilenetv2: [MobileNetV2-Inverted-Residuals-and-Linear-Bottlenecks.pdf](reference/MobileNetV2-Inverted-Residuals-and-Linear-Bottlenecks.pdf)

##### 1.2.2 preprocess

1-crop method is adapted，resize to (N, 256) or (256, N), central crop: 224，mean value：[103.94, 116.78, 123.68], input scale is 0.017.

```Python
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
```

##### 1.2.3 test results

|model name | top1 | top5|
|------------------ | ------- | -------|
|Mobilenetv1 | 0.6989 | 0.8938 |
|Mobilenetv2 | 0.71616 | 0.90226 |



#### 1.3 Inception

> Repo links：
>
> - [caffe-googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
> - [GeekLiB-caffe-model](https://github.com/GeekLiB/caffe-model)

##### 1.3.1 paper

- Inceptionv1: [Going-Deeper-with-Convolutions.pdf](reference/Inceptionv1-Going-Deeper-with-Convolutions.pdf)
- Inceptionv2: [Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift.pdf](reference/Inceptionv2-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift.pdf)
- Inceptionv3: [Rethinking-the-Inception-Architecture-for-Computer-Vision.pdf](reference/Inceptionv3-Rethinking-the-Inception-Architecture-for-Computer-Vision.pdf)
- Inceptionv4: [Inception-v4-Inception-ResNet-and-the-Impact-of-Residual-Connections-on-Learning.pdf](reference/Inception-v4-Inception-ResNet-and-the-Impact-of-Residual-Connections-on-Learning.pdf)

##### 1.3.2 preprocess

1-crop method is adapted:

- inceptionv1：mean value：[104, 117, 123], resize: 256*256, central crop：224
- inceptionv3：mean value：[128, 128, 128, 128], resize: 395*395, central crop: 395, input_scale:  0.0078125(1/128.0) validation dataset：[ILSVRC2015_val](data/ILSVRC2012/val_2015.txt)
- inceptionv4：mean value：[128, 128, 128, 128], resize: 320*320, central crop: 299, input_scale:  0.0078125(1/128.0)

```python
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
```

##### 1.3.3 test results

| model name  | top1    | top5    |
| ----------- | ------- | ------- |
| Inceptionv1 | 0.68528 | 0.88848 |
| Inceptionv3 | 0.7936  | 0.9492  |
| Inceptionv4 | 0.7994  | 0.9502  |



#### 1.4 Squeezenet

> Repo link：[forresti-SqueezeNet](https://github.com/forresti/SqueezeNet)

##### 1.4.1 paper

Squeezenet: [SqueezeNet-AlexNet-level-accuracy-with-50x-fewer-parameters-and-<0.5MB-model-size.pdf](reference/SqueezeNet-AlexNet-level-accuracy-with-50x-fewer-parameters-and-<0.5MB-model-size.pdf)

##### 1.4.2 preprocess

1-crop method is adapted: mean value：[104, 117, 123], resize: 256*256, crop：224

```python
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
```

##### 1.4.3 test results

| model name      | top1    | top5    |
| --------------- | ------- | ------- |
| Squeezenet-v1.0 | 0.57106 | 0.80004 |
| Squeezenet-v1.1 | 0.5771  | 0.80498 |



#### 1.5 Resnet

> Repo link: [KaimingHe repo](https://github.com/KaimingHe/deep-residual-networks)

##### 1.5.1 paper
Resnet：[Deep Residual Learning for Image Recognition.pdf](reference/Resnet-Deep-Residual-Learning-for-Image-Recognition.pdf)

##### 1.5.2 preprocess
1-crop method is adapted: resize to (N, 256) or (256, N), central crop: 224, mean value：[resnet-mean](models/caffe/resnet/ResNet_mean.npy)

```Python
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
```
##### 1.5.3 test results

|model name|top1|top5|
|-|-|-|
|Resnet50|0.7497|0.92088|
|Resnet101|0.76238|0.9286|
|Resnet152|0.76686|0.9321|

#### 1.6 Densenet

> Repo link： [shicai-DenseNet-Caffe](https://github.com/shicai/DenseNet-Caffe)

##### 1.6.1 paper

Densenet: [Densely-Connected-Convolutional-Networks.pdf](reference/Densely-Connected-Convolutional-Networks.pdf)

##### 1.6.2 preprocess

1-crop method is adapted：resize to (N, 256) or (256, N), central crop: 224, mean value: [103.94, 116.78, 123.68], input scale is 0.017.

```Python
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
```

##### 1.6.3 test results

| model name  | top1    | top5    |
| ----------- | ------- | ------- |
| Densenet121 | 0.74898 | 0.92234 |
| Densenet161 | 0.77702 | 0.93826 |
| Densenet169 | 0.76194 | 0.93166 |
| Densenet201 |         |         |



#### 1.7 VGG

> Repo link： [vgg](http://www.robots.ox.ac.uk/~vgg/research/very_deep/#pub)

##### 1.7.1 paper

Vgg: [VGG-Very-Deep-Convolutional-Networks-for-Large-Scale-Image.pdf)](reference/VGG-Very-Deep-Convolutional-Networks-for-Large-Scale-Image.pdf)

##### 1.7.2 preprocess

1-crop method is adapted, resize to (N, 256) or (256, N),  central crop: 224, mean value：[103.939, 116.779, 123.68]

```Python
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
```

##### 1.7.3 test results

| model name | top1    | top5    |
| ---------- | ------- | ------- |
| Vgg16      | 0.71264 | 0.90062 |
| Vgg19      | 0.71248 | 0.89974 |




### 2 Detection Models
#### 2.1 Faster RCNN
#### 2.2 SSD
#### 2.3 RFCN
#### 2.4 YOLO

### 3 Segmentation Models
#### 3.1 FCN
#### 3.2 ENet