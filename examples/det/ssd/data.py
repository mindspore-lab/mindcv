import inspect
import os

import cv2
import numpy as np
from utils import jaccard_numpy, ssd_bboxes_encode

import mindspore.dataset as de


def _rand(a=0.0, b=1.0):
    """Generate random."""
    return np.random.rand() * (b - a) + a


def random_sample_crop(image, boxes):
    """Random Crop the image and boxes"""
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])

    if min_iou is None:
        return image, boxes

    # max trails (50)
    for _ in range(50):
        image_t = image

        w = _rand(0.3, 1.0) * width
        h = _rand(0.3, 1.0) * height

        # aspect ratio constraint b/t .5 & 2
        if h / w < 0.5 or h / w > 2:
            continue

        left = _rand() * (width - w)
        top = _rand() * (height - h)

        rect = np.array([int(top), int(left), int(top + h), int(left + w)])
        overlap = jaccard_numpy(boxes, rect)

        # dropout some boxes
        drop_mask = overlap > 0
        if not drop_mask.any():
            continue

        if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
            continue

        image_t = image_t[rect[0] : rect[2], rect[1] : rect[3], :]

        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        # mask in that both m1 and m2 are true
        mask = m1 * m2 * drop_mask

        # have any valid boxes? try again if not
        if not mask.any():
            continue

        # take only matching gt boxes
        boxes_t = boxes[mask, :].copy()

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
        boxes_t[:, :2] -= rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
        boxes_t[:, 2:4] -= rect[:2]

        return image_t, boxes_t
    return image, boxes


def preprocess_fn(img_id, image, box, is_training, args):
    """Preprocess function for dataset."""
    cv2.setNumThreads(2)

    def _infer_data(image, input_shape):
        img_h, img_w, _ = image.shape
        input_h, input_w = input_shape

        image = cv2.resize(image, (input_w, input_h))

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        return img_id, image, np.array((img_h, img_w), np.float32)

    def _data_aug(image, box, is_training, args):
        """Data augmentation function."""
        ih, iw, _ = image.shape
        h, w = args.image_size

        if not is_training:
            return _infer_data(image, args.image_size)

        # Random crop
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw, _ = image.shape

        # Resize image
        image = cv2.resize(image, (w, h))

        # Flip image or not
        flip = _rand() < 0.5
        if flip:
            image = cv2.flip(image, 1, dst=None)

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        box[:, [0, 2]] = box[:, [0, 2]] / ih
        box[:, [1, 3]] = box[:, [1, 3]] / iw

        if flip:
            box[:, [1, 3]] = 1 - box[:, [3, 1]]

        box, label, num_match = ssd_bboxes_encode(box, args)
        return image, box, label, num_match

    return _data_aug(image, box, is_training, args)


def create_ssd_dataset(
    name,
    root,
    shuffle,
    batch_size,
    python_multiprocessing,
    num_parallel_workers,
    drop_remainder,
    args,
    num_shards=1,
    shard_id=0,
    is_training=True,
):
    """Create SSD dataset with MindDataset."""
    if name == "coco":
        if is_training:
            mindrecord_file = os.path.join(root, "train", "coco0")
        else:
            mindrecord_file = os.path.join(root, "val", "coco0")

        ds = de.MindDataset(
            mindrecord_file,
            columns_list=["img_id", "image", "annotation"],
            num_shards=num_shards,
            shard_id=shard_id,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle,
        )

        decode = de.vision.Decode()
        ds = ds.map(operations=decode, input_columns=["image"])
        change_swap_op = de.vision.HWC2CHW()

        # Computed from random subset of ImageNet training images
        normalize_op = de.vision.Normalize(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        color_adjust_op = de.vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)

        def compose_map_func(img_id, image, annotation):
            return preprocess_fn(img_id, image, annotation, is_training, args)

        if is_training:
            output_columns = ["image", "box", "label", "num_match"]
            trans = [color_adjust_op, normalize_op, change_swap_op]
        else:
            output_columns = ["img_id", "image", "image_shape"]
            trans = [normalize_op, change_swap_op]

        # Note: mindspore-2.0 delete the parameter column_order
        sig = inspect.signature(ds.map)
        pass_column_order = False if "kwargs" in sig.parameters else True

        ds = ds.map(
            operations=compose_map_func,
            input_columns=["img_id", "image", "annotation"],
            output_columns=output_columns,
            column_order=output_columns if pass_column_order else None,
            python_multiprocessing=python_multiprocessing,
            num_parallel_workers=num_parallel_workers,
        )
        if not pass_column_order:
            ds = ds.project(columns=output_columns)

        ds = ds.map(
            operations=trans,
            input_columns=["image"],
            python_multiprocessing=python_multiprocessing,
            num_parallel_workers=num_parallel_workers,
        )
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        return ds
    else:
        raise NotImplementedError
