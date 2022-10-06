import sys

sys.path.append('.')

import pytest

from mindcv.models.model_factory import create_model
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model
from mindspore.communication import init, get_rank, get_group_size

from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.models.registry import list_models
from config import parse_args

MAX=6250
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('in_channels', [3])
@pytest.mark.parametrize('pretrained', [True, False])
@pytest.mark.parametrize('num_classes', [1, 100, MAX])
@pytest.mark.parametrize('checkpoint_path', [None])
def test_model_factory_parallel(mode, num_classes, 
    in_channels, 
    pretrained, checkpoint_path):

    ms.set_context(mode=mode)

    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)

    batch_size = 1
    model_names = list_models()
    for model_name in model_names:
        if checkpoint_path != '':
            pretrained = False
        network = create_model(model_name=model_name,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            pretrained=pretrained,
                            checkpoint_path=checkpoint_path)
        
            # create dataset
        dataset_eval = create_dataset(
            name='ImageNet',
            root='/home/mindspore/dataset/imagenet2012/imagenet/imagenet_original',
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
        iterator = loader_eval.create_dict_iterator()
        data = iterator.__next__()
        result = model.predict(data['image'])
        print(result.shape)
        assert result.shape[0] == batch_size
        assert result.shape[1] == num_classes
