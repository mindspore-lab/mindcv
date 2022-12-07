import os
import sys
import collections
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


def test_repeated_aug():
    
    distribute = False
    #ms.set_context(mode=ms.PYNATIVE_MODE)
    if distribute:
        from mindspore.communication import init, get_rank, get_group_size
        ms.set_context(mode=ms.GRAPH_MODE)
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = 1
        rank_id = 0

    name = 'imagenet'
    '''
    data_dir = '/data/imagenette2-320'
    num_classes = 10
    '''
    dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    root_dir = "./"

    if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
        DownLoad().download_and_extract_archive(dataset_url, root_dir)
    data_dir = "./data/Canidae/"
    num_classes = 2

    num_aug_repeats = 3

    dataset = create_dataset(
        name=name,
        root=data_dir,
        split='val',
        shuffle=True,
        num_samples=None,
        num_parallel_workers=8,
        num_shards=device_num,
        shard_id=rank_id,
        download=False,
        num_aug_repeats=num_aug_repeats
    )

    # load dataset
    loader = create_loader(
        dataset=dataset,
        batch_size=32,
        drop_remainder=True,
        is_training=False,
        transform=None,
        num_classes=num_classes,
        num_parallel_workers=2,
    )
    for epoch in range(1):
        #cnt = 1
        for batch, (data, label) in enumerate(loader.create_tuple_iterator()):
            mean_vals = data.mean(axis=[1,2,3])
            #print(mean_vals, mean_vals.shape)
            rounded = [int(val * 10e8) for val in mean_vals] 
            rep_ele = [item for item, count in collections.Counter(rounded).items() if count > 1]
            #print('repeated instance indices: ', len(rep_ele)) #, rep_ele)
            assert len(rep_ele) > 0, 'Not replicated instances found in the batch'
            if batch == 0:
                print('Epoch: ', epoch, 'Batch: ', batch, 'Rank: ', rank_id, 'Label: ', label[:4])


if __name__=='__main__':
    test_repeated_aug()
