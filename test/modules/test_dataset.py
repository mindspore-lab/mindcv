from pickletools import uint8
import sys
sys.path.append('.')

import pytest

from mindcv.data import create_dataset
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size


# test imagenet
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['ImageNet'])
@pytest.mark.parametrize('split', ['train', 'val'])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('num_samples', [2, 8, None])
@pytest.mark.parametrize('num_parallel_workers', [2, 4, 8, 16])
def test_create_dataset_standalone_imagenet(mode, name, split, shuffle, num_samples, num_parallel_workers):
    '''
    test create_dataset API(standalone)
    command: pytest -s test_dataset.py::test_create_dataset_standalone_imagenet

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
    root = '/data0/dataset/imagenet2012/imagenet_original/'
    
    dataset = create_dataset(
        name=name,
        root=root,
        split=split,
        shuffle=shuffle,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        download=False
    )

    assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None
    print(dataset.output_types())


# test MNIST CIFAR10
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['MNIST', 'CIFAR10'])
@pytest.mark.parametrize('split', ['train', 'val'])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('num_samples', [2, 8, None])
@pytest.mark.parametrize('num_parallel_workers', [2, 4, 8, 16])
@pytest.mark.parametrize('download', [True, False])
def test_create_dataset_standalone_mc(mode, name, split, shuffle, num_samples, num_parallel_workers, download):
    '''
    test create_dataset API(standalone)
    command: pytest -s test_dataset.py::test_create_dataset_standalone_mc

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
    
    dataset = create_dataset(
        name=name,
        split=split,
        shuffle=shuffle,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        download=download
    )

    assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None
    print(dataset.output_types())
