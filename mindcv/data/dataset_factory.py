# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dataset Factory"""

from typing import Optional
import os

from .dataset_download import MnistDownload, Cifar10Download, Cifar100Download

from mindspore.dataset import MnistDataset, Cifar10Dataset, Cifar100Dataset, ImageFolderDataset

_MINDSPORE_BASIC_DATASET = dict(
    mnist=(MnistDataset, MnistDownload),
    cifar10=(Cifar10Dataset, Cifar10Download),
    cifar100=(Cifar100Dataset, Cifar100Download)
)


def create_dataset(
        name,
        root,
        split: str = 'train',
        shuffle: Optional[bool] = None,
        sampler=None,
        num_shards: Optional[int] = None,
        shard_id: Optional[int] = None,
        num_parallel_workers: Optional[int] = None,
        download: bool = False,
        **kwargs
):
    name = name.lower()
    mindspore_kwargs = dict(shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id,
                            num_parallel_workers=num_parallel_workers, **kwargs)
    if name in _MINDSPORE_BASIC_DATASET:
        dataset_class = _MINDSPORE_BASIC_DATASET[name][0]
        dataset_download = _MINDSPORE_BASIC_DATASET[name][1]
        dataset_new_path = None
        if download:
            dataset_download = dataset_download(root)
            dataset_download.download()
            dataset_new_path = dataset_download.path

        dataset = dataset_class(dataset_dir=dataset_new_path if dataset_new_path else root,
                                usage=split,
                                **mindspore_kwargs)

    elif name == 'image_folder' or name == 'folder' or name == 'imagenet':
        if download:
            raise ValueError("Dataset download is not supported.")
        if os.path.isdir(root):
            root = os.path.join(root, split)
        dataset = ImageFolderDataset(dataset_dir=root, **mindspore_kwargs)
    else:
        assert False, "Unknown dataset"

    return dataset
