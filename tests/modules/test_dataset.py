import os
import sys
sys.path.append('.')

import pytest

from mindcv.data import create_dataset
from mindcv.utils.download import DownLoad
import mindspore as ms

# test imagenet
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['ImageNet'])
@pytest.mark.parametrize('split', ['train', 'val'])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('num_samples', [2, None])
@pytest.mark.parametrize('num_parallel_workers', [2])
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
    dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    root_dir = "./"

    if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
        DownLoad().download_and_extract_archive(dataset_url, root_dir)
    data_dir = "./data/Canidae/"
    dataset = create_dataset(
        name=name,
        root=data_dir,
        split=split,
        shuffle=shuffle,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        download=False
    )

    assert type(dataset) == ms.dataset.engine.datasets_vision.ImageFolderDataset
    assert dataset != None


# test MNIST CIFAR10
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['MNIST', 'CIFAR10'])
@pytest.mark.parametrize('split', ['train', 'test'])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('num_samples', [2, None])
@pytest.mark.parametrize('num_parallel_workers', [2])
@pytest.mark.parametrize('download', [True])
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

    assert type(dataset) == ms.dataset.engine.datasets_vision.MnistDataset or \
           type(dataset) == ms.dataset.engine.datasets_vision.Cifar10Dataset
    assert dataset != None
