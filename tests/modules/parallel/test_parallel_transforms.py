import sys
sys.path.append('../..')

import pytest

from mindcv.data import create_dataset, create_transforms, create_loader
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['ImageNet'])
@pytest.mark.parametrize('image_resize', [224, 256, 320])
@pytest.mark.parametrize('is_training', [True, False])
def test_transforms_distribute_imagenet(mode, name, image_resize, is_training):
    '''
    test transform_list API(distribute)
    command: mpirun -n 8 pytest -s test_transforms.py::test_transforms_distribute_imagenet

    API Args:
        dataset_name='',
        image_resize=224,
        is_training=False,
        **kwargs
    '''
    ms.set_context(mode=mode)

    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)

    root = '/data0/dataset/imagenet2012/imagenet_original/'
    dataset = create_dataset(
        name=name,
        root=root,
        split='train',
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=8,
        download=False
    )

    # create transforms
    transform_list = create_transforms(
        dataset_name=name,
        image_resize=image_resize,
        is_training=is_training
    )

    # load dataset
    loader = create_loader(
        dataset=dataset,
        batch_size=32,
        drop_remainder=True,
        is_training=is_training,
        transform=transform_list,
        num_parallel_workers=8,
    )

    print(loader)
    print(loader.output_shapes())

    assert loader.output_shapes()[0][2] == image_resize, 'image_resize error !'


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['MNIST', 'CIFAR10'])
@pytest.mark.parametrize('image_resize', [224, 256, 320])
@pytest.mark.parametrize('is_training', [True, False])
@pytest.mark.parametrize('download', [True, False])
def test_transforms_distribute_imagenet_mc(mode, name, image_resize, is_training, download):
    '''
    test transform_list API(distribute)
    command: mpirun -n 8 pytest -s test_transforms.py::test_transforms_distribute_imagenet_mc

    API Args:
        dataset_name='',
        image_resize=224,
        is_training=False,
        **kwargs
    '''
    ms.set_context(mode=mode)

    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)

    dataset = create_dataset(
        name=name,
        split='train',
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=8,
        download=download
    )

    # create transforms
    transform_list = create_transforms(
        dataset_name=name,
        image_resize=image_resize,
        is_training=is_training
    )

    # load dataset
    loader = create_loader(
        dataset=dataset,
        batch_size=32,
        drop_remainder=True,
        is_training=is_training,
        transform=transform_list,
        num_parallel_workers=8,
    )

    print(loader)
    print(loader.output_shapes())

    assert loader.output_shapes()[0][2] == image_resize, 'image_resize error !'
