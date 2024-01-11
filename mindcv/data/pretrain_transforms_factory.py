"""
Transform operation for pre-training
"""

from typing import List, Tuple, Union

from mindspore.dataset import vision
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import Inter

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .mask_generator import create_mask_generator

__all__ = ["create_transforms_pretrain"]


class RandomResizedCropWithTwoResolution:
    def __init__(self, resize_list: List, interpolations: Union[List, Tuple], scale=(0.08, 1.0), ratio=(0.75, 1.333)):
        self.first_transform = vision.RandomResizedCrop(resize_list[0], scale, ratio, interpolations[0])
        self.second_transform = vision.RandomResizedCrop(resize_list[1], scale, ratio, interpolations[1])

    def __call__(self, img):
        return self.first_transform(img), self.second_transform(img)


class TransformsForPretrain:
    def __init__(
        self,
        resize_list: List = [224],
        tokenizer: str = "dall-e",
        mask_type: str = "block-wise",
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333),
        hflip=0.5,
        color_jitter=None,
        interpolations: Union[List, Tuple] = ["bicubic", "bilinear"],  # lanczos is not implemented in MindSpore
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        patch_size: int = 16,
        mask_ratio: float = 0.4,
        **kwargs
    ):
        for i in range(len(interpolations)):
            if hasattr(Inter, interpolations[i].upper()):
                interpolations[i] = getattr(Inter, interpolations[i].upper())
            else:
                interpolations[i] = Inter.BILINEAR

        if len(resize_list) == 2:
            common_transform = [vision.Decode()]
            if color_jitter is not None:
                if isinstance(color_jitter, (list, tuple)):
                    # color jitter shoulf be a 3-tuple/list for brightness/contrast/saturation
                    # or 4 if also augmenting hue
                    assert len(color_jitter) in (3, 4)
                else:
                    color_jitter = (float(color_jitter),) * 3
                common_transform += [vision.RandomColorAdjust(*color_jitter)]

            if hflip > 0.0:
                common_transform += [vision.RandomHorizontalFlip(prob=hflip)]

            common_transform += [RandomResizedCropWithTwoResolution(resize_list, interpolations, scale, ratio)]
            self.common_transform = Compose(common_transform)

            self.patch_transform = Compose([vision.Normalize(mean=mean, std=std), vision.HWC2CHW()])

            if tokenizer == "dall_e":  # beit
                self.visual_token_transform = Compose([vision.ToTensor(), lambda x: (1 - 2 * 0.1) * x + 0.1])
            elif tokenizer == "vqkd":  # beit v2
                self.visual_token_transform = Compose([vision.ToTensor()])
            elif tokenizer == "clip":  # eva, eva-02
                self.visual_token_transform = Compose(
                    [
                        vision.ToTensor(),
                        vision.Normalize(
                            mean=(0.48145466, 0.4578275, 0.40821073),
                            std=(0.26862954, 0.26130258, 0.27577711),
                            is_hwc=False,
                        ),
                    ]
                )

            self.masked_position_generator = create_mask_generator(
                mask_type, input_size=resize_list[0], patch_size=patch_size, mask_ratio=mask_ratio, **kwargs
            )

            self.output_columns = ["patch", "token", "mask"]
        else:
            self.common_transform = None

            patch_transform = [
                vision.RandomCropDecodeResize(
                    size=resize_list[0], scale=scale, ratio=ratio, interpolation=interpolations[0]
                )
            ]

            if hflip > 0.0:
                patch_transform += [vision.RandomHorizontalFlip(hflip)]

            patch_transform += [vision.Normalize(mean=mean, std=std), vision.HWC2CHW()]
            self.patch_transform = Compose(patch_transform)

            self.masked_position_generator = create_mask_generator(
                mask_type, input_size=resize_list[0], patch_size=patch_size, mask_ratio=mask_ratio, **kwargs
            )

            self.output_columns = ["patch", "mask"]

    def __call__(self, image):
        if self.common_transform is not None:  # for beit, beit v2, eva, eva-02
            patches, visual_tokens = self.common_transform(image)
            patches = self.patch_transform(patches)
            visual_tokens = self.visual_token_transform(visual_tokens)
            masks = self.masked_position_generator()
            return patches, visual_tokens, masks
        else:
            patches = self.patch_transform(image)  # for MAE, SimMIM
            masks = self.masked_position_generator()
            return patches, masks


def create_transforms_pretrain(dataset_name="", **kwargs):
    if dataset_name in ("imagenet", ""):
        return TransformsForPretrain(**kwargs)
    else:
        raise NotImplementedError()
