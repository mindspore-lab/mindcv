from .block_wise_mask import BlockWiseMaskGenerator
from .patch_aligned_mask import PatchAlignedMaskGenerator
from .random_mask import RandomMaskGenerator

__all__ = ["create_mask_generator"]


def create_mask_generator(
    mask_name: str, input_size: int = 224, patch_size: int = 16, mask_ratio: float = 0.6, **kwargs
):
    if mask_name == "random":
        mask_generator = RandomMaskGenerator(input_size, patch_size, mask_ratio)
    elif mask_name == "block_wise":
        mask_generator = BlockWiseMaskGenerator(input_size, patch_size, mask_ratio)
    elif mask_name == "patch_aligned":
        mask_generator = PatchAlignedMaskGenerator(input_size, patch_size, mask_ratio, **kwargs)
    else:
        raise NotImplementedError(f"{mask_name} mask generator is not implemented.")

    return mask_generator
