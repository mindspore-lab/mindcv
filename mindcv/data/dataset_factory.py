from typing import Optional
import os

from .dataset_download import MnistDownload, Cifar10Download, Cifar100Download

from mindspore.dataset import MnistDataset, Cifar10Dataset, Cifar100Dataset, ImageFolderDataset
import mindspore.dataset as ds

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
    '''
    name: dataset name, if empty '' or non-standard dataset name, then process as customized dataset.    
    root: dataset root dir. 
    split: subfolder of root dir, e.g., train, val, test
    shuffle: whether to shuffle the dataset, default: True
    num_samples (int, optional): Number of elements to sample (default=None, which means sample all elements).
    
    For custom datasets and imagenet, the dataset dir should follow the structure like: 
    .dataset_name/
    ├── train/  
    │  ├── class1/
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ....
    │  └── class2/
    │      ├── 000001.jpg
    │      ├── 000002.jpg
    │      └── ....
    └── val/   
       ├── class1/
       │   ├── 000001.jpg
       │   ├── 000002.jpg
       │   └── ....
       └── class2/
           ├── 000001.jpg
           ├── 000002.jpg
           └── ....

    '''
    
    if num_samples == None:
        sampler = None
    elif num_samples > 0:
        if shuffle:
            sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
        else:
            sampler = ds.SequentialSampler(num_samples=num_samples)
        shuffle = None # shuffle and sampler cannot be set at the same in mindspore datatset API
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
                root = os.path.join(root, 'dataset_{}'.format(str(shard_id)))
            dataset_download = dataset_download(root)
            dataset_download.download()
            dataset_new_path = dataset_download.path

        dataset = dataset_class(dataset_dir=dataset_new_path if dataset_new_path else root,
                                usage=split,
                                **mindspore_kwargs)
    else:
        if name == "imagenet" and download:
            raise ValueError("Imagenet dataset download is not supported.")

        if os.path.isdir(root):
            root = os.path.join(root, split)
        dataset = ImageFolderDataset(dataset_dir=root, **mindspore_kwargs)

    return dataset
