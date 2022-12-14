"""
Create dataset by name
"""

from typing import Optional
import os

from mindspore.dataset import MnistDataset, Cifar10Dataset, Cifar100Dataset, ImageFolderDataset, DistributedSampler
import mindspore.dataset as ds

from .dataset_download import MnistDownload, Cifar10Download, Cifar100Download
from .distributed_sampler import RepeatAugSampler
#from .dataset_reader import ImageNetDataset

__all__ = ["create_dataset"]

_MINDSPORE_BASIC_DATASET = dict(
    mnist=(MnistDataset, MnistDownload),
    cifar10=(Cifar10Dataset, Cifar10Download),
    cifar100=(Cifar100Dataset, Cifar100Download)
)

#_DATASET_SIZE = {'imagenet': }

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
        num_aug_repeats: int = 0,
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
        num_aug_repeats: Number of dataset repeatition for repeated augmentation. If 0 or 1, repeated augmentation is diabled. Otherwise, repeated augmentation is enabled and the common choice is 3. (Default: 0)

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

    assert (num_samples is None) or (num_aug_repeats==0), 'num_samples and num_aug_repeats can NOT be set together.'

    name = name.lower()
    # subset sampling
    if num_samples is not None and num_samples > 0:
        # TODO: rewrite ordered distributed sampler
        if num_shards is not None and num_shards > 1: # distributed
            print('ns', num_shards, 'num_samples', num_samples)
            sampler = DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
        else: # standalone
            if shuffle:
                sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
            else:
                sampler = ds.SequentialSampler(num_samples=num_samples)
        mindspore_kwargs = dict(shuffle=None, sampler=sampler,
                            num_parallel_workers=num_parallel_workers, **kwargs)
    else:
        sampler = None
        mindspore_kwargs = dict(shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id,
                            num_parallel_workers=num_parallel_workers, **kwargs)

    # sampler for repeated augmentation
    if num_aug_repeats > 0:
        dataset_size = get_dataset_size(name, root, split)
        print(f'INFO: Repeated augmentation is enabled, num_aug_repeats: {num_aug_repeats}, original dataset size: ', dataset_size)
        #since drop_remainder is usally True, we don't need to do rounding in sampling
        sampler = RepeatAugSampler(dataset_size, num_shards=num_shards, rank_id=shard_id, num_repeats=num_aug_repeats, selected_round=0, shuffle=shuffle)
        mindspore_kwargs = dict(shuffle=None, sampler=sampler, num_shards=None, shard_id=None, **kwargs)

    # create dataset
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
        ''' Another implementation which a bit slower than ImageFolderDataset
            imagenet_dataset = ImageNetDataset(dataset_dir=root)
            # TODO: why round by 256?
            sampler = RepeatAugSampler(len(imagenet_dataset), num_shards=num_shards, rank_id=shard_id, num_repeats=repeated_aug, selected_round=1, shuffle=shuffle)
            dataset = ds.GeneratorDataset(imagenet_dataset, column_names=imagenet_dataset.column_names, sampler=sampler)
        '''
    return dataset

def get_dataset_size(name, root, split):
    if name in _MINDSPORE_BASIC_DATASET:
        dataset_class = _MINDSPORE_BASIC_DATASET[name][0]
        dataset = dataset_class(dataset_dir=root, usage=split)
    else:
        if os.path.isdir(root):
            root = os.path.join(root, split)
        dataset = ImageFolderDataset(dataset_dir=root)

    return dataset.get_dataset_size()
