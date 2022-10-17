"""
Create dataset by name
"""

from typing import Optional
import os

from mindspore.dataset import MnistDataset, Cifar10Dataset, Cifar100Dataset, ImageFolderDataset
import mindspore.dataset as ds

from .dataset_download import MnistDownload, Cifar10Download, Cifar100Download

__all__ = ["create_dataset"]

_MINDSPORE_BASIC_DATASET = dict(
    mnist=(MnistDataset, MnistDownload),
    cifar10=(Cifar10Dataset, Cifar10Download),
    cifar100=(Cifar100Dataset, Cifar100Download)
)


def create_dataset(
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
):
    r"""Creates dataset by name.

    Args:
        name: dataset name like MNIST, CIFAR10, ImageNeT, ''. '' means a customized dataset. Default: ''.
        root: dataset root dir. Default: './'.
        split: data split: '' or split name string (train/val/test), if it is '', no split is used.
            Otherwise, it is a subfolder of root dir, e.g., train, val, test. Default: 'train'.
        shuffle: whether to shuffle the dataset. Default: True.
        num_samples: Number of elements to sample (default=None, which means sample all elements).
        num_shards: Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id: The shard ID within `num_shards` (default=None).
            This argument can only be specified when `num_shards` is also specified.
        num_parallel_workers: Number of workers to read the data (default=None, set in the config).
        download: whether to download the dataset. Default: False

    Note:
        For custom datasets and imagenet, the dataset dir should follow the structure like:
        .dataset_name/
        ├── split1/
        │  ├── class1/
        │  │   ├── 000001.jpg
        │  │   ├── 000002.jpg
        │  │   └── ....
        │  └── class2/
        │      ├── 000001.jpg
        │      ├── 000002.jpg
        │      └── ....
        └── split2/
           ├── class1/
           │   ├── 000001.jpg
           │   ├── 000002.jpg
           │   └── ....
           └── class2/
               ├── 000001.jpg
               ├── 000002.jpg
               └── ....

    Returns:
        Dataset object
    """

    if num_samples is None:
        sampler = None
    elif num_samples > 0:
        if shuffle:
            sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
        else:
            sampler = ds.SequentialSampler(num_samples=num_samples)
        shuffle = None  # shuffle and sampler cannot be set at the same in mindspore datatset API
    else:
        sampler = None

    name = name.lower()
    mindspore_kwargs = dict(shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id,
                            num_parallel_workers=num_parallel_workers, **kwargs)
    if name in _MINDSPORE_BASIC_DATASET:
        dataset_class = _MINDSPORE_BASIC_DATASET[name][0]
        dataset_download = _MINDSPORE_BASIC_DATASET[name][1]
        dataset_new_path = None
        if download:
            if shard_id is not None:
                root = os.path.join(root, f'dataset_{str(shard_id)}')
            dataset_download = dataset_download(root)
            dataset_download.download()
            dataset_new_path = dataset_download.path

        dataset = dataset_class(dataset_dir=dataset_new_path if dataset_new_path else root,
                                usage=split,
                                **mindspore_kwargs)
        # address ms dataset num_classes empty issue
        if name == 'mnist':
            dataset.num_classes = lambda :10
        elif name == 'cifar10':
            dataset.num_classes = lambda :10
        elif name == 'cifar100':
            dataset.num_classes = lambda :100

    else:
        if name == "imagenet" and download:
            raise ValueError("Imagenet dataset download is not supported.")

        if os.path.isdir(root):
            root = os.path.join(root, split)
        dataset = ImageFolderDataset(dataset_dir=root, **mindspore_kwargs)

    return dataset
