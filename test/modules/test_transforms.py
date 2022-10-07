import os
import sys
sys.path.append('.')

import pytest

from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.utils.download import DownLoad
import mindspore as ms


# test imagenet
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['ImageNet'])
@pytest.mark.parametrize('image_resize', [224, 256])
@pytest.mark.parametrize('is_training', [True, False])
def test_transforms_standalone_imagenet(mode, name, image_resize, is_training):
    '''
    test transform_list API(distribute)
    command: pytest -s test_transforms.py::test_transforms_standalone_imagenet

    API Args:
        dataset_name='',
        image_resize=224,
        is_training=False,
        **kwargs
    '''
    ms.set_context(mode=mode)

    dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    root_dir = "./"

    if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
        DownLoad().download_and_extract_archive(dataset_url, root_dir)
    data_dir = "./data/Canidae/"
    dataset = create_dataset(
        name=name,
        root=data_dir,
        split='train',
        shuffle=True,
        num_samples=None,
        num_parallel_workers=2,
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
        num_parallel_workers=2,
    )

    assert loader.output_shapes()[0][2] == image_resize, 'image_resize error !'


# test mnist cifar10
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['MNIST', 'CIFAR10'])
@pytest.mark.parametrize('image_resize', [224, 256])
@pytest.mark.parametrize('is_training', [True, False])
@pytest.mark.parametrize('download', [True])
def test_transforms_standalone_dataset_mc(mode, name, image_resize, is_training, download):
    '''
    test transform_list API(distribute)
    command: pytest -s test_transforms.py::test_transforms_standalone_imagenet_mc

    API Args:
        dataset_name='',
        image_resize=224,
        is_training=False,
        **kwargs
    '''
    ms.set_context(mode=mode)
    
    dataset = create_dataset(
        name=name,
        split='train',
        shuffle=True,
        num_samples=None,
        num_parallel_workers=2,
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
        num_parallel_workers=2,
    )

    assert loader.output_shapes()[0][2] == image_resize, 'image_resize error !'


# test is_training
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['ImageNet'])
@pytest.mark.parametrize('image_resize', [224, 256])
def test_transforms_standalone_imagenet_is_training(mode, name, image_resize):
    '''
    test transform_list API(distribute)
    command: pytest -s test_transforms.py::test_transforms_standalone_imagenet_is_training

    API Args:
        dataset_name='',
        image_resize=224,
        is_training=False,
        **kwargs
    '''
    ms.set_context(mode=mode)

    # create transforms
    transform_list_train = create_transforms(
        dataset_name=name,
        image_resize=image_resize,
        is_training=True
    )
    transform_list_val = create_transforms(
        dataset_name=name,
        image_resize=image_resize,
        is_training=False
    )

    assert type(transform_list_train) == list
    assert type(transform_list_val) == list
    assert transform_list_train != transform_list_val
