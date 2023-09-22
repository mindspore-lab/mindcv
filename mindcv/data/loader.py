"""
Create dataloader
"""

import inspect
import warnings

import numpy as np

import mindspore as ms
from mindspore.dataset import transforms

from .mixup import Mixup
from .transforms_factory import create_transforms

__all__ = ["create_loader"]


def create_loader(
    dataset,
    batch_size,
    drop_remainder=False,
    is_training=False,
    mixup=0.0,
    cutmix=0.0,
    cutmix_prob=0.0,
    num_classes=1000,
    transform=None,
    target_transform=None,
    num_parallel_workers=None,
    python_multiprocessing=False,
    separate=False,
):
    r"""Creates dataloader.

    Applies operations such as transform and batch to the `ms.dataset.Dataset` object
    created by the `create_dataset` function to get the dataloader.

    Args:
        dataset (ms.dataset.Dataset): dataset object created by `create_dataset`.
        batch_size (int or function): The number of rows each batch is created with. An
            int or callable object which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether to drop the last block
            whose data row number is less than batch size (default=False). If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        is_training (bool): whether it is in train mode. Default: False.
        mixup (float): mixup alpha, mixup will be enabled if > 0. (default=0.0).
        cutmix (float): cutmix alpha, cutmix will be enabled if > 0. (default=0.0). This operation is experimental.
        cutmix_prob (float): prob of doing cutmix for an image (default=0.0)
        num_classes (int): the number of classes. Default: 1000.
        transform (list or None): the list of transformations that wil be applied on the image,
            which is obtained by `create_transform`. If None, the default imagenet transformation
            for evaluation will be applied. Default: None.
        target_transform (list or None): the list of transformations that will be applied on the label.
            If None, the label will be converted to the type of ms.int32. Default: None.
        num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel
            (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
            option could be beneficial if the Python operation is computational heavy (default=False).
        separate(bool, optional): separate the image clean and the image been transformed.
            If separate==True, that means the dataset returned has 3 parts:
            * the first part called image "clean", which means the image without auto_augment (e.g., auto-aug)
            * the second and third parts called image transformed, hence, with the auto_augment transform.
            Refer to ".transforms_factory.create_transforms" for more information.

    Note:
        1. cutmix is now experimental (which means performance gain is not guarantee)
            and can not be used together with mixup due to the label int type conflict.
        2. `is_training`, `mixup`, `num_classes` is used for MixUp, which is a kind of transform operation.
          However, we are not able to merge it into `transform`, due to the limitations of the `mindspore.dataset` API.


    Returns:
        BatchDataset, dataset batched.
    """

    if target_transform is None:
        target_transform = transforms.TypeCast(ms.int32)
    target_input_columns = "label" if "label" in dataset.get_col_names() else "fine_label"
    dataset = dataset.map(
        operations=target_transform,
        input_columns=target_input_columns,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    if transform is None:
        warnings.warn(
            "Using None as the default value of transform will set it back to "
            "traditional image transform, which is not recommended. "
            "You should explicitly call `create_transforms` and pass it to `create_loader`."
        )
        transform = create_transforms("imagenet", is_training=False)

    # only apply augment splits to train dataset
    if separate and is_training:
        assert isinstance(transform, tuple) and len(transform) == 3

        # Note: mindspore-2.0 delete the parameter column_order
        sig = inspect.signature(dataset.map)
        pass_column_order = False if "kwargs" in sig.parameters else True

        # map all the transform
        dataset = map_transform_splits(
            dataset, transform, num_parallel_workers, python_multiprocessing, pass_column_order
        )
        # after batch, datasets has 4 columns
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        # concat the 3 columns of image
        dataset = dataset.map(
            operations=concat_per_batch_map,
            input_columns=["image_clean", "image_aug1", "image_aug2", "label"],
            output_columns=["image", "label"],
            column_order=["image", "label"] if pass_column_order else None,
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=python_multiprocessing,
        )

    else:
        dataset = dataset.map(
            operations=transform,
            input_columns="image",
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=python_multiprocessing,
        )

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    if is_training:
        if (mixup + cutmix > 0.0) and batch_size > 1:
            # TODO: use mindspore vision cutmix and mixup after the confliction fixed in later release
            # set label_smoothing 0 here since label smoothing is computed in loss module
            mixup_fn = Mixup(
                mixup_alpha=mixup,
                cutmix_alpha=cutmix,
                cutmix_minmax=None,
                prob=cutmix_prob,
                switch_prob=0.5,
                label_smoothing=0.0,
                num_classes=num_classes,
            )
            # images in a batch are mixed. labels are converted soft onehot labels.
            dataset = dataset.map(
                operations=mixup_fn,
                input_columns=["image", target_input_columns],
                num_parallel_workers=num_parallel_workers,
            )

    return dataset


def map_transform_splits(dataset, transform, num_parallel_workers, python_multiprocessing, pass_column_order):
    # map the primary_tfl such as to all the images
    dataset = dataset.map(
        operations=transform[0],
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    # duplicate the columns 'image' twice for the auto_augmentation
    dataset = dataset.map(
        operations=transforms.Duplicate(),
        input_columns=["image"],
        output_columns=["image_clean", "image_aug2"],
        column_order=["image_clean", "image_aug2", "label"] if pass_column_order else None,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )
    dataset = dataset.map(
        operations=transforms.Duplicate(),
        input_columns=["image_clean"],
        output_columns=["image_clean", "image_aug1"],
        column_order=["image_clean", "image_aug1", "image_aug2", "label"] if pass_column_order else None,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    # map the secondary_tfl (auto_augmentation for the image_aug1 and img_aug2)
    dataset = dataset.map(
        operations=transform[1],
        input_columns="image_aug1",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    dataset = dataset.map(
        operations=transform[1],
        input_columns="image_aug2",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    # map the final_tfl to all the images

    dataset = dataset.map(
        operations=transform[2],
        input_columns="image_clean",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    dataset = dataset.map(
        operations=transform[2],
        input_columns="image_aug1",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    dataset = dataset.map(
        operations=transform[2],
        input_columns="image_aug2",
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
    )

    return dataset


def concat_per_batch_map(image_clean, image_aug1, image_aug2, label):
    image = np.concatenate((image_clean, image_aug1, image_aug2))
    label = np.concatenate((label, label, label))
    return image, label
