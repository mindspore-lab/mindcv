from pickletools import uint8
import sys
sys.path.append('.')

import pytest

from mindcv.data import create_dataset
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['ImageNet'])
@pytest.mark.parametrize('split', ['train', 'val'])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('num_parallel_workers', [2, 4, 8, 16])
def test_create_dataset_distribute_imagenet(mode, name, split, shuffle, num_parallel_workers):
    '''
    test create_dataset API(distribute)
    command: mpirun -n 8 pytest -s test_dataset.py::test_create_dataset_distribute_imagenet

    API Args:
        name: str = '',
        root: str = './',
        split: str = 'train',
        shuffle: bool = True,
        num_samples: Optional[bool] = None,
        num_shards: Optional[int] = None,
        shard_id: Optional[int] = None,
        num_parallel_workers: Optional[int] = None,
        download: bool = False,
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
        split=split,
        shuffle=shuffle,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=num_parallel_workers,
        download=False
    )

    assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None
    print(dataset.output_types())


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['MNIST', 'CIFAR10'])
@pytest.mark.parametrize('split', ['train', 'val'])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('num_parallel_workers', [2, 4, 8, 16])
@pytest.mark.parametrize('download', [True, False])
def test_create_dataset_distribute_mc(mode, name, split, shuffle, num_parallel_workers, download):
    '''
    test create_dataset API(distribute)
    command: mpirun -n 8 pytest -s test_dataset.py::test_create_dataset_distribute_mc

    API Args:
        name: str = '',
        root: str = './',
        split: str = 'train',
        shuffle: bool = True,
        num_samples: Optional[bool] = None,
        num_shards: Optional[int] = None,
        shard_id: Optional[int] = None,
        num_parallel_workers: Optional[int] = None,
        download: bool = False,
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
        split=split,
        shuffle=shuffle,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=num_parallel_workers,
        download=download
    )

    assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None
    print(dataset.output_types())
