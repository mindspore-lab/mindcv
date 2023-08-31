import cv2
import numpy as np

import mindspore.dataset as de

cv2.setNumThreads(0)


class SegDataset:
    def __init__(
        self,
        image_mean,
        image_std,
        data_file="",
        batch_size=32,
        crop_size=512,
        max_scale=2.0,
        min_scale=0.5,
        ignore_label=255,
        num_classes=21,
        num_readers=2,
        num_parallel_calls=4,
        shard_id=None,
        shard_num=None,
        shuffle=True,
    ):
        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        self.shuffle = shuffle
        assert max_scale > min_scale

    def get_dataset(self):
        data_set = de.MindDataset(
            dataset_files=self.data_file,
            columns_list=["data", "label"],
            shuffle=True,
            num_parallel_workers=self.num_readers,
            num_shards=self.shard_num,
            shard_id=self.shard_id,
        )

        input_columns = ["data", "label"]
        output_columns = ["data", "label"]
        transforms_list = self.train_preprocess_

        data_set = data_set.map(
            operations=transforms_list,
            input_columns=input_columns,
            output_columns=output_columns,
            num_parallel_workers=self.num_parallel_calls,
        )

        if self.shuffle:
            data_set = data_set.shuffle(buffer_size=self.batch_size * 10)

        data_set = data_set.batch(self.batch_size, drop_remainder=True)

        return data_set


def create_segment_dataset(
    name,
    data_dir,
    is_training=False,
    args=None,
):
    if name == "voc" or name == "vocaug":
        if is_training:
            dataset = SegDataset(
                image_mean=args.image_mean,
                image_std=args.image_std,
                data_file=data_dir,
                batch_size=args.batch_size,
                crop_size=args.crop_size,
                max_scale=args.max_scale,
                min_scale=args.min_scale,
                ignore_label=args.ignore_label,
                num_classes=args.num_classes,
                num_readers=args.num_parallel_workers,
                num_parallel_calls=args.num_parallel_workers,
                shard_id=args.shard_id,
                shard_num=args.shard_num,
                shuffle=args.shuffle,
            )
            return dataset.get_dataset()

        else:
            with open(data_dir) as f:
                eval_data_list = f.readlines()
            return eval_data_list

    else:
        raise NotImplementedError
