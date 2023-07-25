""" Read images online """
import numpy as np

from mindspore.dataset import GeneratorDataset


class ReadDataset:
    def __init__(self, annotation_dir, images_dir):
        # Read annotations
        self.annotation = {}
        for i in open(annotation_dir, "r"):
            image_label = i.replace("\n", "").replace("/", "_").split(" ")
            image = image_label[0] + ".jpg"
            label = " ".join(image_label[1:])
            self.annotation[image] = label

        # Transfer string-type label to int-type label
        self.label2id = {}
        labels = sorted(list(set(self.annotation.values())))
        for i in labels:
            self.label2id[i] = labels.index(i)

        for image, label in self.annotation.items():
            self.annotation[image] = self.label2id[label]

        # Read image-labels as iterable object
        images = dict.fromkeys(self.label2id.values(), [])
        for image, label in self.annotation.items():
            read_image = np.fromfile(images_dir + image, dtype=np.uint8)
            images[label].append(read_image)

        self._data = sum(list(images.values()), [])
        self._label = sum([[i] * len(images[i]) for i in images.keys()], [])

    # make class ReadDataset an iterable object
    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


# take aircraft dataset as an example
annotation_dir = "./aircraft/data/images_variant_trainval.txt"
images_dir = "./aircraft/data/iamges/"
dataset = ReadDataset(annotation_dir)
dataset_train = GeneratorDataset(source=dataset, column_names=["image", "label"], shuffle=True)
