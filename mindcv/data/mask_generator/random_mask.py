import numpy as np


class RandomMaskGenerator:
    def __init__(self, input_size: int = 224, model_patch_size: int = 16, mask_ratio: float = 0.75):
        assert input_size % model_patch_size == 0

        self.grid_size = input_size // model_patch_size
        self.seq_len = self.grid_size**2
        self.mask_count = int(np.ceil(self.seq_len * mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.seq_len)[: self.mask_count]
        mask = np.zeros(self.seq_len, dtype=np.int32)
        mask[mask_idx] = 1

        mask = mask.reshape((self.grid_size, self.grid_size))
        return mask
