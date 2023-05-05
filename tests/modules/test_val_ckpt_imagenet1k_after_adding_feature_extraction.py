"""
Test feature extraction and validation accuracy with checkpoint on
ImageNet-1K after adding feature extraction for ResNet, MobileNetV3, ConvNeXt,
ResNeST, EfficientNet, RepVGG, ReXNet.

Please specify 'imagenet1k_root' before running the test.
"""
import os
import sys

import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Model

sys.path.append("../..")

from mindcv.data import create_dataset, create_loader, create_transforms
from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.utils import check_batch_size
from mindcv.utils.download import DownLoad

from config import parse_args  # isort: skip

imagenet1k_root = '/path/to/ImageNet-1K/'


def output_feature(args):
    network = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        pretrained=args.pretrained,
        checkpoint_path=args.ckpt_path,
        ema=args.ema,
        features_only=True
    )

    x = ms.Tensor(np.random.randn(8, 3, 32, 32), dtype=ms.float32)
    out = network(x)

    return len(out)


def validate_with_ckpt_imagenet1k(args):
    ms.set_context(mode=args.mode)

    # create dataset
    dataset_eval = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.val_split,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download,
    )

    # create transform
    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=False,
        image_resize=args.image_resize,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
    )

    # read num clases
    num_classes = dataset_eval.num_classes() if args.num_classes is None else args.num_classes

    # check batch size
    batch_size = check_batch_size(dataset_eval.get_dataset_size(), args.batch_size)

    # load dataset
    loader_eval = create_loader(
        dataset=dataset_eval,
        batch_size=batch_size,
        drop_remainder=False,
        is_training=False,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

    # create model
    network = create_model(
        model_name=args.model,
        num_classes=num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        pretrained=args.pretrained,
        checkpoint_path=args.ckpt_path,
        ema=args.ema,
    )

    # for params in network.get_parameters():
    #     print('Name: ', params.name)

    assert isinstance(network, nn.Cell), "Loading checkpoint error"

    network.set_train(False)

    # create loss
    loss = create_loss(
        name=args.loss,
        reduction=args.reduction,
        label_smoothing=args.label_smoothing,
        aux_factor=args.aux_factor,
    )

    # Define eval metrics.
    eval_metrics = {
        "Top_1_Accuracy": nn.Top1CategoricalAccuracy(),
        "loss": nn.metrics.Loss(),
    }

    # init model
    model = Model(network, loss_fn=loss, metrics=eval_metrics)

    # validate
    result = model.eval(loader_eval, dataset_sink_mode=False)

    return result


@pytest.mark.parametrize('config, ckpt_path, ckpt_link, data_dir, acc_target, length_target',
                         [('../../configs/resnet/resnet_18_ascend.yaml',
                           '../../checkpoints/resnet/resnet18-1e65cd21.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt',
                           imagenet1k_root, 0.7, 5),
                          ('../../configs/mobilenetv3/mobilenet_v3_small_ascend.yaml',
                           '../../checkpoints/mobilenetv3/mobilenet_v3_small_100-c884b105.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv3/mobilenet_v3_small_100-c884b105.ckpt',
                           imagenet1k_root, 0.67, 5),
                          ('../../configs/convnext/convnext_tiny_ascend.yaml',
                           '../../checkpoints/convnext/convnext_tiny-ae5ff8d7.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_tiny-ae5ff8d7.ckpt',
                           imagenet1k_root, 0.8, 4),
                          ('../../configs/resnest/resnest50_ascend.yaml',
                           '../../checkpoints/resnest/resnest50-f2e7fc9c.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/resnest/resnest50-f2e7fc9c.ckpt',
                           imagenet1k_root, 0.8, 5),
                          ('../../configs/efficientnet/efficientnet_b0_ascend.yaml',
                           '../../checkpoints/efficientnet/efficientnet_b0-103ec70c.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/efficientnet/efficientnet_b0-103ec70c.ckpt',
                           imagenet1k_root, 0.76, 5),
                          ('../../configs/repvgg/repvgg_a0_ascend.yaml',
                           '../../checkpoints/repvgg/repvgg_a0-6e71139d.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/repvgg/repvgg_a0-6e71139d.ckpt',
                           imagenet1k_root, 0.72, 5),
                          ('../../configs/rexnet/rexnet_x10_ascend.yaml',
                           '../../checkpoints/rexnet/rexnet_10-c5fb2dc7.ckpt',
                           'https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_10-c5fb2dc7.ckpt',
                           imagenet1k_root, 0.77, 5)])
def test_val_ckpt_imagenet1k(config, ckpt_path, ckpt_link, data_dir, acc_target, length_target):
    [ckpt_dir, ckpt_name] = os.path.split(ckpt_path)
    ckpt_root = os.sep.join(ckpt_dir.split(os.sep)[:-1])

    if not os.path.isdir(ckpt_root):
        os.mkdir(ckpt_root)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.isfile(ckpt_path):
        DownLoad().download_url(ckpt_link, ckpt_dir, ckpt_name)

    args = ['-c', config, '--ckpt_path', ckpt_path, '--data_dir', data_dir]
    args = parse_args(args)

    length = output_feature(args)
    assert length == length_target, "Wrong feature extraction"

    result = validate_with_ckpt_imagenet1k(args)
    assert result['Top_1_Accuracy'] > acc_target, "Top_1_Accuracy is abnormal"
