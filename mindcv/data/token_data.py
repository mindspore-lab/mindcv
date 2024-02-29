import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from mindspore.dataset import vision
from mindspore.dataset.transforms import Compose

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

_logger = logging.getLogger(__name__)

ALLOWED_FORMAT = {".jpeg", ".jpg", ".bmp", ".png"}


def create_img_transform(is_train: bool = True):
    if is_train:
        transform_list = [
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            vision.HWC2CHW(),
        ]
    else:
        transform_list = [
            vision.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            vision.HWC2CHW(),
        ]
    transform = Compose(transform_list)
    return transform


def _cal_size_train(w: int, h: int, max_size: int, min_size: int) -> Tuple[int, int]:
    def mapping(x):  # linear [-1, 1] -> [min, max]
        return (max_size - min_size) / 2 * x + (max_size + min_size) / 2

    u = np.random.normal(-0.5, 1)
    u = np.clip(u, -1, 1)
    target = mapping(u)

    # sample the a random side to [min_size, maxsize],
    # resize the another side without change the aspect ratio (most of the cases)
    if np.random.random() > 0.5:
        new_w = target
        new_h = np.clip(h / w * target, min_size, max_size)
    else:
        new_h = target
        new_w = np.clip(w / h * target, min_size, max_size)
    return new_w, new_h


def _cal_size_infer(w: int, h: int, max_size: int, min_size: int) -> Tuple[int, int]:
    if max(w, h) > max_size:
        if w < h:
            new_h = max_size
            new_w = w / h * max_size
        else:
            new_w = max_size
            new_h = h / w * max_size
    elif min(w, h) < min_size:
        if w < h:
            new_w = min_size
            new_h = h / w * min_size
        else:
            new_h = min_size
            new_w = w / h * min_size
    else:
        new_w, new_h = w, h

    new_h = np.clip(new_h, min_size, max_size)
    new_w = np.clip(new_w, min_size, max_size)
    return new_w, new_h


def _cal_size(
    shape: Tuple[int, int], patch_size: int = 32, max_size: int = 384, min_size: int = 64, is_train: bool = False
):
    w, h = shape

    if is_train:  # resolution sampling
        new_w, new_h = _cal_size_train(w, h, max_size, min_size)
    else:  # keep in range without change aspect ratio
        new_w, new_h = _cal_size_infer(w, h, max_size, min_size)

    target_shape = np.maximum(np.round(np.array([new_w, new_h]) / patch_size), 1) * patch_size
    target_shape = target_shape.astype(int)
    return target_shape.tolist()


class ImageTokenDataset:
    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 16,
        max_seq_length: int = 2048,
        enable_cache: bool = True,
        cache_path: str = "train_db_info.json",
        interpolation: str = "bilinear",
        image_resize: int = 384,
        image_resize_min: int = 64,
        max_num_each_group: int = 32,
    ) -> None:
        self.is_train = split == "train"
        self.patch_size = patch_size
        self.max_seq_length = max_seq_length
        self.image_resize = image_resize
        self.image_resize_min = image_resize_min
        self.max_num_each_group = max_num_each_group

        self.transform = create_img_transform(is_train=self.is_train)
        if interpolation == "bilinear":
            self.resample = Image.BILINEAR
        elif interpolation == "bicubic":
            self.resample = Image.BICUBIC
        else:
            raise ValueError(f"Unsupported interpolation `{interpolation}`")

        root = os.path.join(root, split)

        # step 1: search all image under root and store there original image size
        # and target image size (which is divisible by patch size).
        self.images_info = self._inspect_images(root, enable_cache=enable_cache, cache_path=cache_path)

        # step 2: group the images by the maximum sequence length
        # here we use greedy method for simplicty, and meanwhile we do the resolution sampling in the same time
        self.update_groups()

        # step 3 (optional): form the label mapping for train/val, for imagenet format
        label = sorted(list(set([x["label"] for x in self.images_info])))
        self.label_mapping = dict(zip(label, range(len(label))))

    def update_groups(self):
        # call this once after each epoch allowing different resolution sampling
        self.images_group = self._group_by_max_seq_length(self.images_info)
        max_num_each_group_real = max([len(x) for x in self.images_group])
        if max_num_each_group_real > self.max_num_each_group:
            _logger.warning(
                f"The Maximum number of images in the grouping ({max_num_each_group_real}) "
                f"is higher than the allowed value ({self.max_num_each_group}), "
                "you may need to adjust the `max_num_each_group` in the model and dataloader."
            )

        _logger.info(f"Packing group is updated. The total number of groups now is {len(self.images_group)}.")

    def __len__(self):
        return len(self.images_group)

    def __getitem__(self, index):
        if self.is_train:
            try:
                img_group = self.images_group[index]
            except (
                IndexError
            ):  # the total number of image groups may change after group update, but __len__ may not not changed.
                img_group = random.choice(self.images_group)
        else:
            img_group = self.images_group[index]

        img_patch_seq = list()
        pos_seq = list()
        label_seq = list()
        for img_info in img_group:
            with Image.open(img_info["img_path"]) as f:
                img = f.convert("RGB")

            img_shape = list(img.size)
            if img_shape != img_info["shape"]:
                raise RuntimeError(
                    f"The image shape `{img_shape}` is different from shape in cache `{img_info['shape']}`. "
                    "Perhaps the cache is corrupted, please remove the cache first and rerun again.",
                )

            # step 1: resize to the shape which is divisible by patch size
            img = img.resize(img_info["target_shape"], resample=self.resample)

            # stes 2: normalization and other imagewise transform
            img = self.transform(img)[0]

            # step 3: patchify
            img_patch, pos = self._patchify(img)
            img_patch_seq.append(img_patch)
            pos_seq.append(pos)

            # step 4 (optinal): add label
            label = self.label_mapping[img_info["label"]]
            label_seq.append(label)

        # create the image index within each batch, use to generate image-level mask
        img_patch_index = [i * np.ones(len(x), dtype=np.int32) for i, x in enumerate(img_patch_seq)]

        img_patch_seq = np.concatenate(img_patch_seq, dtype=np.float32)
        pos_seq = np.concatenate(pos_seq, dtype=np.int32)
        img_patch_index = np.concatenate(img_patch_index, dtype=np.int32)
        label_index = np.array(label_seq, dtype=np.int32)
        return img_patch_seq, pos_seq, img_patch_index, label_index

    def _inspect_images(
        self, root: str, enable_cache: bool = True, cache_path: str = "train_db_info.json"
    ) -> List[Dict[str, Any]]:
        images_info = list()

        if enable_cache:
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                _logger.warning(f"Reading image info from `{cache_path}` is failed. ({str(e)})")

        _logger.info(f"Scanning images under `{root}`.")
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in ALLOWED_FORMAT:
                    label = os.path.basename(dirpath)
                    fpath = os.path.join(dirpath, f)
                    with Image.open(fpath) as f:
                        shape = f.size

                    images_info.append(
                        {
                            "img_path": fpath,
                            "label": label,
                            "shape": list(shape),
                        }
                    )

        images_info = sorted(images_info, key=lambda x: x["img_path"])
        if enable_cache:
            _logger.info(f"Save image info to `{cache_path}`.")
            with open(cache_path, "w") as f:
                json.dump(images_info, f, indent=4)

        return images_info

    def _group_by_max_seq_length(self, images_info: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups, group = list(), list()
        seq_len = 0

        if self.is_train:
            random.shuffle(images_info)

        for image_info in images_info:
            w, h = _cal_size(
                image_info["shape"],
                patch_size=self.patch_size,
                max_size=self.image_resize,
                min_size=self.image_resize_min,
                is_train=self.is_train,
            )
            nw, nh = w // self.patch_size, h // self.patch_size
            img_seq_len = nw * nh

            image_info["target_shape"] = (w, h)

            if img_seq_len > self.max_seq_length:
                _logger.warning(
                    f"single image sequence length ({img_seq_len}) with image shape `{w, h}` "
                    f"is longer than the max sequence length ({self.max_seq_length}), skip."
                )
                continue

            if (seq_len + img_seq_len) > self.max_seq_length or len(group) > self.max_num_each_group:
                groups.append(group)
                group = list()
                seq_len = 0

            group.append(image_info)
            seq_len += img_seq_len

        # the last
        groups.append(group)
        return groups

    def _patchify(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c, h, w = img.shape
        nh, nw = h // self.patch_size, w // self.patch_size

        img = np.reshape(img, (c, nh, self.patch_size, nw, self.patch_size))
        img = np.transpose(img, (1, 3, 0, 2, 4))  # nh, nw, c, patch, patch
        img = np.reshape(img, (nh * nw, -1))  # nh * nw, c * patch * patch

        # get 2d abs posiition
        pos = np.meshgrid(np.arange(nh), np.arange(nw), indexing="ij")
        pos = np.stack(pos, axis=-1)
        pos = np.reshape(pos, (-1, 2))
        return img, pos
