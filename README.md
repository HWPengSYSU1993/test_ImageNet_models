# test_ImageNet_models
Test classic ImageNet-pretrained image classification models including VGG, Inception and ResNet
### Environment

- python==3.6.5
- tensorflow==1.12.0


### Getting Started

1. Download the required model's .ckpt files and place in the "models" folder.

   | Model                                                  | Checkpoint                                                   | Top-1 Accuracy | Top-5 Accuracy |
   | ------------------------------------------------------ | ------------------------------------------------------------ | -------------- | -------------- |
   | [Inception V1](http://arxiv.org/abs/1409.4842v1)       | [inception_v1_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) | 69.8           | 89.6           |
   | [Inception V2](http://arxiv.org/abs/1502.03167)        | [inception_v2_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz) | 73.9           | 91.8           |
   | [Inception V3](http://arxiv.org/abs/1512.00567)        | [inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) | 78.0           | 93.9           |
   | [Inception V4](http://arxiv.org/abs/1602.07261)        | [inception_v4_2016_09_09.tar.gz](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) | 80.2           | 95.2           |
   | [Inception-ResNet-v2](http://arxiv.org/abs/1602.07261) | [inception_resnet_v2_2016_08_30.tar.gz](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) | 80.4           | 95.3           |
   | [ResNet V1 50](https://arxiv.org/abs/1512.03385)       | [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) | 75.2           | 92.2           |
   | [ResNet V1 101](https://arxiv.org/abs/1512.03385)      | [resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) | 76.4           | 92.9           |
   | [ResNet V2 50](https://arxiv.org/abs/1603.05027)       | [resnet_v2_50_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz) | 75.6           | 92.8           |
   | [ResNet V2 101](https://arxiv.org/abs/1603.05027)      | [resnet_v2_101_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) | 77.0           | 93.7           |
   | [ResNet V2 152](https://arxiv.org/abs/1603.05027)      | [resnet_v2_152_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz) | 77.8           | 94.1           |
   | [VGG 16](http://arxiv.org/abs/1409.1556.pdf)           | [vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) | 71.5           | 89.8           |
   | [VGG 19](http://arxiv.org/abs/1409.1556.pdf)           | [vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) | 71.1           | 89.8           |

2. run "test_single_image.py" to classify image based different models.

