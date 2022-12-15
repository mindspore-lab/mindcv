''' distributed sampler '''
import math
import numpy as np

class RepeatAugSampler():
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process.

    This sampler was adapted from https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py

    Args:
        dataset_size: dataset size
        num_shards: num devices
        rank_id: device id
        shuffle: shuffle
        num_repeats: num of repeated instances in repeated augmentation, default:3
        selected_round: round the total num of samples by this factor
    """
    def __init__(
            self,
            dataset_size,
            num_shards=None,
            rank_id=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
    ):
        if num_shards is None:
            print("WARNING: num_shards is set to 1 in RepeatAugSampler since it is not passed in")
            num_shards = 1
        if rank_id is None:
            rank_id = 0

        #assert isinstance(num_repeats, int), f'num_repeats should be Type integer, but got {type(num_repeats)}'

        self.dataset_size = dataset_size
        self.num_shards = num_shards
        self.rank_id = rank_id
        self.shuffle = shuffle
        self.num_repeats = int(num_repeats)
        self.epoch = 0
        self.num_samples = int(math.ceil(self.dataset_size * num_repeats / self.num_shards))
        self.total_size = self.num_samples * self.num_shards
        # Determine the number of samples to select per epoch for each rank.
        if selected_round:
            self.num_selected_samples = int(math.floor(
                 self.dataset_size // selected_round * selected_round / num_shards))
        else:
            self.num_selected_samples = int(math.ceil(self.dataset_size / num_shards))

    def __iter__(self):
        # deterministically shuffle based on epoch
        #print('__iter__  generating new shuffled indices: ', self.epoch)
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            indices = indices.tolist()
            self.epoch += 1
            #print(indices[:30])
        else:
            indices = list(range(self.dataset_size))
        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        indices = [ele for ele in indices for i in range(self.num_repeats)]

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank_id:self.total_size:self.num_shards]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__=='__main__':
    num_devices = 2
    dataset_size = 20
    num_repeats = 3
    selected_round = 1
    shuffle = True
    sampler1 = RepeatAugSampler(dataset_size, num_shards=num_devices, rank_id=0, num_repeats=num_repeats, selected_round=selected_round, shuffle=shuffle)
    sampler2 = RepeatAugSampler(dataset_size, num_shards=num_devices, rank_id=1, num_repeats=num_repeats, selected_round=selected_round, shuffle=shuffle)
    output1 = list(sampler1.__iter__())
    output2 = list(sampler2.__iter__())
    print(output1)
    print(output2)
    print(len(output1), len(output2), len(output1+output2))
    assert len(output1 + output2) == dataset_size, 'sizes of sampled outputs do not match dataset size'
