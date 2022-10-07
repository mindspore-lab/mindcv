import os
import sys

sys.path.append('.')

import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Model

from mindcv.utils.download import DownLoad
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.models.model_factory import create_model
from mindcv.models.registry import list_models


MAX=6250
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('in_channels', [3])
@pytest.mark.parametrize('pretrained', [False])
@pytest.mark.parametrize('num_classes', [2])
def test_create_model_standalone(mode, num_classes, in_channels, pretrained):
    batch_size = 1
    ms.set_context(mode=mode)
    model_names = list_models()
    dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    root_dir = "./"

    if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
        DownLoad().download_and_extract_archive(dataset_url, root_dir)
    data_dir = "./data/Canidae/"
    for model_name in model_names:
        network = create_model(model_name=model_name,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            pretrained=pretrained)
        
            # create dataset
        dataset_eval = create_dataset(
            name='ImageNet',
            root=data_dir,
            split='val',
            num_samples=1,
            num_parallel_workers=1,
            download=False)

        # create transform
        transform_list = create_transforms(
            dataset_name='ImageNet',
            is_training=False,
            image_resize=224,
            interpolation='bilinear',
        )

        # load dataset
        loader_eval = create_loader(
            dataset=dataset_eval,
            batch_size=batch_size,
            drop_remainder=False,
            is_training=False,
            transform=transform_list,
            num_parallel_workers=1,
        )

        # create model
        network = network
        network.set_train(False)

        # create loss
        loss = create_loss(name='CE', 
                        label_smoothing=0.1, 
                        aux_factor=0) 

        # Define eval metrics.
        eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                        'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

        # init model
        model = Model(network, loss_fn=loss, metrics=eval_metrics)
        # result = model.eval(loader_eval)
        iterator = loader_eval.create_dict_iterator()
        data = iterator.__next__()
        result = model.predict(data['image'])
        print(result.shape)
        assert result.shape[0] == batch_size
        assert result.shape[1] == num_classes
