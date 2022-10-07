import os
import sys
sys.path.append('.')

import pytest

from mindcv.data import create_dataset, create_loader
from mindcv.utils.download import DownLoad
import mindspore as ms
from mindspore.dataset.transforms import OneHot


num_classes = 1
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('split', ['train'])
@pytest.mark.parametrize('batch_size', [1, 16])
@pytest.mark.parametrize('drop_remainder', [True, False])
@pytest.mark.parametrize('is_training', [True, False])
@pytest.mark.parametrize('transform', [None])
@pytest.mark.parametrize('target_transform', [None, [OneHot(num_classes)]])
@pytest.mark.parametrize('mixup', [0, 1])
@pytest.mark.parametrize('num_classes', [2])
@pytest.mark.parametrize('num_parallel_workers', [None, 2])
@pytest.mark.parametrize('python_multiprocessing', [True, False])
def test_dataset_loader_standalone(mode, split, batch_size, drop_remainder, is_training,
                                   transform, target_transform, mixup, num_classes,
                                   num_parallel_workers, python_multiprocessing):
    '''
    test create_dataset API(standalone)
    command: pytest -s test_dataset.py::test_create_dataset_standalone
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
    name = 'ImageNet'
    if name == 'ImageNet':
        dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
        root_dir = "./"

        if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
            DownLoad().download_and_extract_archive(dataset_url, root_dir)
        data_dir = "./data/Canidae/"

    dataset = create_dataset(
        name=name,
        root=data_dir,
        split=split,
        shuffle=False,
        num_parallel_workers=num_parallel_workers,
        download=False
    )

    # load dataset
    loader_train = create_loader(
        dataset=dataset,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        is_training=is_training,
        transform=transform,
        target_transform=target_transform,
        mixup=mixup,
        num_classes=num_classes,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )
    out_batch_size = loader_train.get_batch_size()
    out_shapes = loader_train.output_shapes()[0]
    assert out_batch_size == batch_size
    assert out_shapes == [batch_size, 3, 224, 224]