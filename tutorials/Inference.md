# Image Classification Prediction

This tutorial introduces how to call the pretraining model in MindCV to make classification prediction on the test image.

## Model Loading

### View All Available Models

By calling the `registry.list_models` function in `mindcv.models`, the names of all network models can be printed. The models of a network in different parameter configurations will also be printed, such as resnet18 / resnet34 / resnet50 / resnet101 / resnet152.


```python
import sys
sys.path.append("..")
from mindcv.models import registry
registry.list_models()
```




    ['BiTresnet50',
     'RepMLPNet_B224',
     'RepMLPNet_B256',
     'RepMLPNet_D256',
     'RepMLPNet_L256',
     'RepMLPNet_T224',
     'RepMLPNet_T256',
     'convit_base',
     'convit_base_plus',
     'convit_small',
     'convit_small_plus',
     'convit_tiny',
     'convit_tiny_plus',
     'convnext_base',
     'convnext_large',
     'convnext_small',
     'convnext_tiny',
     'convnext_xlarge',
     'densenet121',
     'densenet161',
     'densenet169',
     'densenet201',
     'dpn107',
     'dpn131',
     'dpn92',
     'dpn98',
     'efficientnet_b0',
     'efficientnet_b1',
     'efficientnet_b2',
     'efficientnet_b3',
     'efficientnet_b4',
     'efficientnet_b5',
     'efficientnet_b6',
     'efficientnet_b7',
     'efficientnet_v2_l',
     'efficientnet_v2_m',
     'efficientnet_v2_s',
     'efficientnet_v2_xl',
     'ghostnet_1x',
     'ghostnet_nose_1x',
     'googlenet',
     'inception_v3',
     'inception_v4',
     'mnasnet0_5',
     'mnasnet0_75',
     'mnasnet1_0',
     'mnasnet1_3',
     'mnasnet1_4',
     'mobilenet_v1_025_224',
     'mobilenet_v1_050_224',
     'mobilenet_v1_075_224',
     'mobilenet_v1_100_224',
     'mobilenet_v2_035_128',
     'mobilenet_v2_035_160',
     'mobilenet_v2_035_192',
     'mobilenet_v2_035_224',
     'mobilenet_v2_035_96',
     'mobilenet_v2_050_128',
     'mobilenet_v2_050_160',
     'mobilenet_v2_050_192',
     'mobilenet_v2_050_224',
     'mobilenet_v2_050_96',
     'mobilenet_v2_075_128',
     'mobilenet_v2_075_160',
     'mobilenet_v2_075_192',
     'mobilenet_v2_075_224',
     'mobilenet_v2_075_96',
     'mobilenet_v2_100_128',
     'mobilenet_v2_100_160',
     'mobilenet_v2_100_192',
     'mobilenet_v2_100_224',
     'mobilenet_v2_100_96',
     'mobilenet_v2_130_224',
     'mobilenet_v2_140_224',
     'mobilenet_v3_large_075',
     'mobilenet_v3_large_100',
     'mobilenet_v3_small_075',
     'mobilenet_v3_small_100',
     'nasnet',
     'pnasnet',
     'poolformer_m36',
     'poolformer_m48',
     'poolformer_s12',
     'poolformer_s24',
     'poolformer_s36',
     'pvt_huge_v2',
     'pvt_large',
     'pvt_medium',
     'pvt_small',
     'pvt_tiny',
     'regnet_x_12gf',
     'regnet_x_16gf',
     'regnet_x_1_6gf',
     'regnet_x_200mf',
     'regnet_x_32gf',
     'regnet_x_3_2gf',
     'regnet_x_400mf',
     'regnet_x_4_0gf',
     'regnet_x_600mf',
     'regnet_x_6_4gf',
     'regnet_x_800mf',
     'regnet_x_8_0gf',
     'regnet_y_12gf',
     'regnet_y_16gf',
     'regnet_y_1_6gf',
     'regnet_y_200mf',
     'regnet_y_32gf',
     'regnet_y_3_2gf',
     'regnet_y_400mf',
     'regnet_y_4_0gf',
     'regnet_y_600mf',
     'regnet_y_6_4gf',
     'regnet_y_800mf',
     'regnet_y_8_0gf',
     'repvgg',
     'res2net101',
     'res2net101_v1b',
     'res2net152',
     'res2net152_v1b',
     'res2net50',
     'res2net50_v1b',
     'resnet101',
     'resnet152',
     'resnet18',
     'resnet34',
     'resnet50',
     'resnext101_32x4d',
     'resnext101_64x4d',
     'resnext152_64x4d',
     'resnext50_32x4d',
     'rexnet_x09',
     'rexnet_x10',
     'rexnet_x13',
     'rexnet_x15',
     'rexnet_x20',
     'shufflenet_v1_g3_x0_5',
     'shufflenet_v1_g3_x1_0',
     'shufflenet_v1_g3_x1_5',
     'shufflenet_v1_g3_x2_0',
     'shufflenet_v1_g8_x0_5',
     'shufflenet_v1_g8_x1_0',
     'shufflenet_v1_g8_x1_5',
     'shufflenet_v1_g8_x2_0',
     'shufflenet_v2_x0_5',
     'shufflenet_v2_x1_0',
     'shufflenet_v2_x1_5',
     'shufflenet_v2_x2_0',
     'sk_resnet18',
     'sk_resnet34',
     'sk_resnet50',
     'sk_resnext50_32x4d',
     'squeezenet1_0',
     'squeezenet1_1',
     'swin_tiny',
     'vgg11',
     'vgg13',
     'vgg16',
     'vgg19',
     'visformer_small',
     'visformer_small_v2',
     'visformer_tiny',
     'visformer_tiny_v2',
     'vit_b_16_224',
     'vit_b_16_384',
     'vit_b_32_224',
     'vit_b_32_384',
     'vit_l_16_224',
     'vit_l_16_384',
     'vit_l_32_224',
     'xception']



### Load Pretraining Model

Taking the resnet50 model as an example, we introduce two methods to load the model checkpoint using the `create_model` function in `mindcv.models`. 1). When the `pretrained` parameter in the interface is set to True, network weights can be automatically downloaded.


```python
from mindcv.models import create_model
model = create_model(model_name='resnet50', num_classes=1000, pretrained=True)
# Switch the execution logic of the network to the inference scenario
model.set_train(False)
```

    102453248B [00:16, 6092186.31B/s]                                                                                      
    




    ResNet<
      (conv1): Conv2d<input_channels=3, output_channels=64, kernel_size=(7, 7), stride=(2, 2), pad_mode=pad, padding=3, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
      (relu): ReLU<>
      (max_pool): MaxPool2d<kernel_size=3, stride=2, pad_mode=SAME>
      (layer1): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=64, output_channels=64, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.bn2.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.bn2.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.bn2.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.bn2.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.bn3.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.bn3.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.bn3.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.bn3.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.down_sample.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.down_sample.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.down_sample.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.down_sample.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=256, output_channels=64, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.1.bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.1.bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.1.bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.1.bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.1.bn2.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.1.bn2.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.1.bn2.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.1.bn2.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.1.bn3.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.1.bn3.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.1.bn3.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.1.bn3.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=256, output_channels=64, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.2.bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.2.bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.2.bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.2.bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.2.bn2.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.2.bn2.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.2.bn2.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.2.bn2.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.2.bn3.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.2.bn3.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.2.bn3.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.2.bn3.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (layer2): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=256, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=256, output_channels=512, kernel_size=(1, 1), stride=(2, 2), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.down_sample.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.down_sample.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.down_sample.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.down_sample.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.1.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.1.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.1.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.1.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.1.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.1.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.1.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.1.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.1.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.1.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.1.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.1.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.2.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.2.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.2.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.2.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.2.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.2.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.2.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.2.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.2.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.2.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.2.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.2.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (3): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.3.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.3.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.3.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.3.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.3.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.3.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.3.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.3.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.3.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.3.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.3.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.3.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (layer3): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=512, output_channels=1024, kernel_size=(1, 1), stride=(2, 2), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.down_sample.1.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.down_sample.1.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.down_sample.1.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.down_sample.1.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.1.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.1.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.1.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.1.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.1.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.1.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.1.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.1.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.1.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.1.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.1.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.1.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.2.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.2.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.2.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.2.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.2.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.2.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.2.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.2.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.2.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.2.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.2.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.2.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (3): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.3.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.3.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.3.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.3.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.3.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.3.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.3.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.3.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.3.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.3.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.3.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.3.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (4): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.4.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.4.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.4.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.4.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.4.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.4.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.4.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.4.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.4.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.4.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.4.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.4.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (5): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.5.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.5.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.5.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.5.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.5.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.5.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.5.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.5.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.5.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.5.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.5.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.5.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (layer4): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.bn1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.bn1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.bn1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.bn1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.bn2.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.bn2.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.bn2.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.bn2.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=512, output_channels=2048, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.bn3.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.bn3.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.bn3.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.bn3.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=1024, output_channels=2048, kernel_size=(1, 1), stride=(2, 2), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.down_sample.1.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.down_sample.1.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.down_sample.1.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.down_sample.1.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=2048, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.1.bn1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.1.bn1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.1.bn1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.1.bn1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.1.bn2.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.1.bn2.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.1.bn2.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.1.bn2.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=512, output_channels=2048, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.1.bn3.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.1.bn3.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.1.bn3.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.1.bn3.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=2048, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.2.bn1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.2.bn1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.2.bn1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.2.bn1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.2.bn2.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.2.bn2.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.2.bn2.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.2.bn2.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=512, output_channels=2048, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.2.bn3.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.2.bn3.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.2.bn3.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.2.bn3.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (pool): GlobalAvgPooling<>
      (classifier): Dense<input_channels=2048, output_channels=1000, has_bias=True>
      >



2). When the `checkpoint_path` parameter in the interface is set to the file path, the model parameter file with the `.ckpt` can be loaded.


```python
from mindcv.models import create_model
model = create_model(model_name='resnet50', num_classes=1000, checkpoint_path='./resnet50_224.ckpt')
# Switch the execution logic of the network to the inference scenario
model.set_train(False)
```




    ResNet<
      (conv1): Conv2d<input_channels=3, output_channels=64, kernel_size=(7, 7), stride=(2, 2), pad_mode=pad, padding=3, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
      (relu): ReLU<>
      (max_pool): MaxPool2d<kernel_size=3, stride=2, pad_mode=SAME>
      (layer1): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=64, output_channels=64, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.bn2.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.bn2.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.bn2.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.bn2.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.bn3.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.bn3.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.bn3.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.bn3.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.0.down_sample.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.0.down_sample.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.0.down_sample.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.0.down_sample.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=256, output_channels=64, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.1.bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.1.bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.1.bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.1.bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.1.bn2.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.1.bn2.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.1.bn2.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.1.bn2.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.1.bn3.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.1.bn3.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.1.bn3.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.1.bn3.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=256, output_channels=64, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.2.bn1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.2.bn1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.2.bn1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.2.bn1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.2.bn2.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.2.bn2.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.2.bn2.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.2.bn2.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=64, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer1.2.bn3.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer1.2.bn3.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer1.2.bn3.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer1.2.bn3.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (layer2): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=256, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=256, output_channels=512, kernel_size=(1, 1), stride=(2, 2), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.0.down_sample.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.0.down_sample.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.0.down_sample.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.0.down_sample.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.1.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.1.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.1.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.1.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.1.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.1.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.1.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.1.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.1.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.1.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.1.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.1.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.2.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.2.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.2.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.2.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.2.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.2.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.2.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.2.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.2.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.2.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.2.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.2.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (3): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=128, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.3.bn1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.3.bn1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.3.bn1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.3.bn1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.3.bn2.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.3.bn2.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.3.bn2.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.3.bn2.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=128, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer2.3.bn3.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer2.3.bn3.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer2.3.bn3.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer2.3.bn3.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (layer3): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=512, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=512, output_channels=1024, kernel_size=(1, 1), stride=(2, 2), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.0.down_sample.1.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.0.down_sample.1.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.0.down_sample.1.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.0.down_sample.1.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.1.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.1.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.1.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.1.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.1.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.1.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.1.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.1.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.1.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.1.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.1.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.1.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.2.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.2.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.2.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.2.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.2.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.2.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.2.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.2.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.2.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.2.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.2.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.2.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (3): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.3.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.3.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.3.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.3.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.3.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.3.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.3.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.3.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.3.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.3.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.3.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.3.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (4): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.4.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.4.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.4.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.4.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.4.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.4.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.4.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.4.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.4.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.4.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.4.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.4.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (5): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=256, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.5.bn1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.5.bn1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.5.bn1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.5.bn1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.5.bn2.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.5.bn2.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.5.bn2.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.5.bn2.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=256, output_channels=1024, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=1024, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer3.5.bn3.gamma, shape=(1024,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer3.5.bn3.beta, shape=(1024,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer3.5.bn3.moving_mean, shape=(1024,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer3.5.bn3.moving_variance, shape=(1024,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (layer4): SequentialCell<
        (0): Bottleneck<
          (conv1): Conv2d<input_channels=1024, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.bn1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.bn1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.bn1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.bn1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.bn2.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.bn2.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.bn2.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.bn2.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=512, output_channels=2048, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.bn3.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.bn3.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.bn3.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.bn3.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          (down_sample): SequentialCell<
            (0): Conv2d<input_channels=1024, output_channels=2048, kernel_size=(1, 1), stride=(2, 2), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (1): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.0.down_sample.1.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.0.down_sample.1.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.0.down_sample.1.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.0.down_sample.1.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
            >
          >
        (1): Bottleneck<
          (conv1): Conv2d<input_channels=2048, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.1.bn1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.1.bn1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.1.bn1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.1.bn1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.1.bn2.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.1.bn2.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.1.bn2.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.1.bn2.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=512, output_channels=2048, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.1.bn3.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.1.bn3.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.1.bn3.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.1.bn3.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        (2): Bottleneck<
          (conv1): Conv2d<input_channels=2048, output_channels=512, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.2.bn1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.2.bn1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.2.bn1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.2.bn1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv2): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn2): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.2.bn2.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.2.bn2.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.2.bn2.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.2.bn2.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
          (conv3): Conv2d<input_channels=512, output_channels=2048, kernel_size=(1, 1), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
          (bn3): BatchNorm2d<num_features=2048, eps=1e-05, momentum=0.9, gamma=Parameter (name=layer4.2.bn3.gamma, shape=(2048,), dtype=Float32, requires_grad=True), beta=Parameter (name=layer4.2.bn3.beta, shape=(2048,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=layer4.2.bn3.moving_mean, shape=(2048,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=layer4.2.bn3.moving_variance, shape=(2048,), dtype=Float32, requires_grad=False)>
          (relu): ReLU<>
          >
        >
      (pool): GlobalAvgPooling<>
      (classifier): Dense<input_channels=2048, output_channels=1000, has_bias=True>
      >



## Data Preparation

### Create Dataset

Here, we download a Wikipedia image as a test image, and use the `create_dataset` function in `mindcv.data` to construct a custom dataset for a single image.


```python
from mindcv.data import create_dataset
num_workers = 1
# path of dataset
data_dir = "./data/"
dataset = create_dataset(root=data_dir, split='test', num_parallel_workers=num_workers)
# Image visualization
from PIL import Image
Image.open("./data/test/dog/dog.jpg")
```




![png](output_8_0.png)



The directory structure of the dataset is as follows:

```Text
data/
└─ test
    ├─ dog
    │   ├─ dog.jpg
    │   └─ ……
    └─ ……
```

### Data Preprocessing

Call the `create_transforms` function to obtain the data processing strategy (transform list) of the ImageNet dataset used by the pre training model.

We pass the obtained transform list into the `create_loader` function, specify `batch_size=1` and other parameters, and then complete the preparation of test data. The `Dataset` object is returned as the input of the model.


```python
from mindcv.data import create_transforms, create_loader
transforms_list = create_transforms(dataset_name='imagenet', is_training=False)
data_loader = create_loader(
        dataset=dataset,
        batch_size=1,
        is_training=False,
        num_classes=1000,
        transform=transforms_list,
        num_parallel_workers=num_workers
    )
```

## Model Inference
The picture of the user-defined dataset is transferred to the model to obtain the inference result. Here, use the `Squeeze` function of `mindspore.ops` to remove the batch dimension.



```python
import mindspore.ops as P
import numpy as np
images, _ = next(data_loader.create_tuple_iterator())
output = P.Squeeze()(model(images))
pred = np.argmax(output.asnumpy())
```


```python
with open("imagenet1000_clsidx_to_labels.txt") as f:
    idx2label = eval(f.read())
print('predict: {}'.format(idx2label[pred]))
```

    predict: Labrador retriever
    
