import numpy as np


class PatchAlignedMaskGenerator:
    def __init__(
        self, input_size: int = 192, model_patch_size: int = 4, mask_ratio: float = 0.6, mask_patch_size: int = 32
    ):
        assert input_size % mask_patch_size == 0
        assert mask_patch_size % model_patch_size == 0

        self.rand_size = input_size // mask_patch_size
        self.scale = mask_patch_size // model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.int32)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask
