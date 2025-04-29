# Fine-tune with A Custom Dataset

This document introduces the process for fine-tuning a pre-trained model from MindCV on a custom dataset and the implementation of fine-tuning techniques such as reading the dataset online, setting the learning rate for specific layers, freezing part of the parameters, etc. The main code is in./example/finetune.py, you can make changes to it based on this tutorial as needed.

Next, we will use the FGVC-Aircraft dataset as an example to show how to fine-tune the pre-trained model mobilenet v3-small. [Fine-Grained Visual Classification of Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a commonly used fine-grained image Classification benchmark dataset, which contains 10,000 aircraft images from 100 different types of aircraft (a.k.a variants), that is, 100 images for each aircraft type.

First, extract the downloaded dataset to . /data folder, the directory structure of the Aircraft dataset is:

```text
aircraft
└── data
    ├── images
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ....
    ├──	images_variant_test.txt
    ├──	images_variant_trainval.txt
    └── ....
```

The folder "images" contains all the 10,000 images, and the airplane types and subset names of each image are recorded in images_variant_*.txt. When this dataset is used for fine-tuning, the training set is usually set by annotation file: images_variant_trainval.txt. Hence, the training set should contain 6667 images and the test set should contain 3333 images after the dataset has been split.

## Data Preprocessing

### Read Custom Dataset

For custom datasets, you can either organize the dataset file directory locally into a tree structure similar to ImageNet, and then use the function `create_dataset` to read the dataset (offline way), or if your dataset is medium-scale or above, which is not suitable to use offline way, you can also directly [read all the images into a mappable or iterable object](https://www.mindspore.cn/tutorials/en/r2.1/beginner/dataset.html#customizing-dataset), replacing the file splitting and the `create_dataset` steps (online way).

#### Read Dataset Offline

The function ` create_dataset ` uses [`mindspore.Dataset.ImageFolderDataset`](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/dataset/mindspore.dataset.ImageFolderDataset.html#mindspore.dataset.ImageFolderDataset) function to build a dataset object, all images in the same folder will be assigned a same label, which is the folder name. Therefore, the prerequisite for using this function is that the file directory of the source dataset should follow the following tree structure:

```text
DATASET_NAME
    ├── split1(e.g. train)/
    │  ├── class1/
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ....
    │  └── class2/
    │      ├── 000001.jpg
    │      ├── 000002.jpg
    │      └── ....
    └── split2/
       ├── class1/
       │   ├── 000001.jpg
       │   ├── 000002.jpg
       │   └── ....
       └── class2/
           ├── 000001.jpg
           ├── 000002.jpg
           └── ....
```

Next, we'll take the annotation file ./aircraft/data/images_variant_trainval.txt as an example, locally generate the file of train set ./aircraft/data/images/trainval/, which meets the request of a tree-structure directory.

```python
import os
import shutil


# only for Aircraft dataset but not a general one
def extract_images(images_path, subset_name, annotation_file_path, copy=True):
    # read the annotation file to get the label of each image
    def annotations(annotation_file_path):
        image_label = {}
        with open(annotation_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                label = " ".join(line.split(" ")[1:]).replace("\n", "").replace("/", "_")
                if label not in image_label.keys():
                    image_label[label] = []
                    image_label[label].append(line.split(" ")[0])
                else:
                    image_label[label].append(line.split(" ")[0])
        return image_label

    # make a new folder for subset
    subset_path = images_path + subset_name
    os.mkdir(subset_path)

    # extract and copy/move images to the new folder
    image_label = annotations(annotation_file_path)
    for label in image_label.keys():
        label_folder = subset_path + "/" + label
        os.mkdir(label_folder)
        for image in image_label[label]:
            image_name = image + ".jpg"
            if copy:
                shutil.copy(images_path + image_name, label_folder + image_name)
            else:
                shutil.move(images_path + image_name, label_folder)


# take train set of aircraft dataset as an example
images_path = "./aircraft/data/images/"
subset_name = "trainval"
annotation_file_path = "./aircraft/data/images_variant_trainval.txt"
extract_images(images_path, subset_name, annotation_file_path)
```

The splitting method of the test set is the same as that of the training set. The file structure of the whole Aircraft dataset should be:

```text
aircraft
└── data
	└── images
        ├── trainval
        │   ├── 707-320
        │   │   ├── 0056978.jpg
        │   │   └── ....
        │   ├── 727-200
        │   │   ├── 0048341.jpg
        │   │	└── ....
        │  	└── ....
        └── test
            ├── 707-320
            │   ├── 0062765.jpg
            │   └── ....
            ├── 727-200
            │   ├── 0061581.jpg
            │   └── ....
            └── ....
```

./example/finetune.py integrates the whole training pipeline, from pre-processing to the establishment and training of the model: `create_dataset` -> `create_transforms` -> `create_loader` -> `create_model` ->..., thus, the dataset with the adequate file directory structure can be sent directly to the fine-tuning script to start the subsequent processes including loading dataset and model training by running `python ./example/finetune.py --data_dir=./aircraft/data/images/` command. For custom datasets, please note that<font color=DarkRed> the dataset parameter in the configuration file must be set to an empty string `""` </font> in advance.

#### Read Dataset Online

Offline data reading takes up extra local disk space to store the newly generated data files. Therefore, when the local storage space is insufficient or the data cannot be backed up to the local environment, the local data files cannot be read using `create_dataset` directly, you can write a function to read the dataset in an online way.

Here's how we generate a random-accessible dataset object that stores the images of the training set and realize a map from indices to data samples:

- First, we define a class `ImageClsDataset` to read the raw data and transform them into a random-accessible dataset:

	- In the initialization function `__init__()`, the annotation file path such as ./aircraft/data/images_variant_trainval.txt is taken as input, and used to generate a dictionary `self.annotation` that stores the one-to-one correspondence between images and tags;
	- Since `create_loader` will perform a map operation on this iterated object, which does not support string format labels, it is also necessary to generate `self.label2id` to<font color=DarkRed> convert the string format label in `self.annotation` to integer type</font>;
	- Based on the information stored in `self.annotation`, we next <font color=DarkRed>read each image in the training set as a one-dimensional array </font> from the folder ./aircraft/data/images/ (the image data must be read as an one-dimensional array due to map operation restrictions in `create_loader`). The image information and label are stored in `self._data` and `self._label` respectively.
	- Next, the mappable object is constructed using the `__getitem__` function.
- After writing the ImageClsDataset class, we can pass it the path of the annotation file to instantiate it, and load it as a dataset that can be read by the model through [`mindspore.dataset.GeneratorDataset`](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset). Note that <font color=DarkRed>the parameter `column_names` must be set to be ["image", "label"]</font> for subsequent reading by other functions. What we've got now is supposed to be the same as what's generated by `create_dataset`.


```python
import numpy as np

from mindspore.dataset import GeneratorDataset


class ImageClsDataset:
    def __init__(self, annotation_dir, images_dir):
        # Read annotations
        self.annotation = {}
        with open(annotation_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_label = line.replace("\n", "").replace("/", "_").split(" ")
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

        # Read image-labels as mappable object
        label2images = {key: [] for key in self.label2id.values()}
        for image, label in self.annotation.items():
            read_image = np.fromfile(images_dir + image, dtype=np.uint8)
            label2images[label].append(read_image)

        self._data = sum(list(label2images.values()), [])
        self._label = sum([[i] * len(label2images[i]) for i in label2images.keys()], [])

    # make class ImageClsDataset a mappable object
    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


# take aircraft dataset as an example
annotation_dir = "./aircraft/data/images_variant_trainval.txt"
images_dir = "./aircraft/data/images/"
dataset = ImageClsDataset(annotation_dir, images_dir)
dataset_train = GeneratorDataset(source=dataset, column_names=["image", "label"], shuffle=True)
```

Compared with the offline way, the online way skipped the step of splitting the data file locally and reading the local file with the `create_dataset` function. So in the subsequent training, simply **replace the part of finetune.py that uses `create_dataset` with the above code**, then you can start training by running finetune.py directly as what you do after reading the dataset offline.

### Augmentation and Batching

MindCV uses the `create_loader` function to perform data augmentation and batching for the dataset read in the previous chapter. Augmentation strategies are defined in advance by the `create_transforms` function. Batching is set by the parameter `batch_size` in the `create_loader` function. All hyperparameters mentioned above can be passed through the model configuration file. Hyper-parameters' specific usage see the [API documentation](https://mindspore-lab.github.io/mindcv/zh/reference/data/).

For small-size custom datasets, it is suggested that data augmentation can be used to the training set to enhance the generalization of the model and prevent overfitting. For the dataset of fine-grained image classification tasks, such as the Aircraft dataset in this tutorial, the classification effect may be not that ideal due to the large variance within the data class, the image size can be set larger by adjusting the hyper-parameter `image_resize` (such as 448, 512, 600, etc.).

## Fine-tuning

Referring to [Stanford University CS231n](https://cs231n.github.io/transfer-learning/#tf), **fine-tuning all the parameters**, **freezing feature network**, and **setting learning rates for specific layers** are commonly used fine-tuning skills. The first one uses pre-trained weights to initialize the parameters of the target model, and then updates all parameters based on the new dataset, so it's usually time-consuming but will get a high precision. Freezing feature networks are divided into freezing all feature networks(linear probe) and freezing partial feature networks. The former uses the pre-trained model as a feature extractor and only updates the parameters of the full connection layer, which takes a short time but has low accuracy; The latter generally freezes the parameters of shallow layers, which only learn the basic features of images, and only updates the parameters of the deep network and the full connection layer. Setting learning rate for specific layers is similar but more elaborate, it specifies the learning rates used by certain layers during training.

For hyper-parameters used in fine-tuning training, you can refer to the configuration file used when pre-trained on the ImageNet-1k dataset in ./configs. Note that for fine-tuning, <font color=DarkRed>the hyper-parameter `pretrained` should be set to be `True` </font>to automatically downloaded and load the pre-trained weight, the parameters of the classification layer will be automatically removed during the process because the `num_classes` is not the default 1000 (but if you want to load a checkpoint file from local directory, do remember to set `pretrained` to be `False`, set the `ckpt_path` and manually delete the parameters of classifier before everything), <font color=DarkRed> `num_classes` should be set to be the number of labels </font>of the custom dataset (e.g. 100 for the Aircraft dataset here), moreover, don't forget to <font color=DarkRed>reduce batch_size and epoch_size </font>based on the size of the custom dataset. In addition, since the pre-trained weight already contains a lot of information for identifying images, in order not to destroy this information too much, it is also necessary to<font color=DarkRed> reduce the learning rate `lr` </font>, and it is also recommended to start training and adjust from at most one-tenth of the pre-trained learning rate or 0.0001. These parameters can be modified in the configuration file or added in the shell command as shown below. The training results can be viewed in the file ./ckpt/results.txt.

```bash
python .examples/finetune/finetune.py --config=./configs/mobilenetv3/mobilnet_v3_small_ascend.yaml --data_dir=./aircraft/data --num_classes=100 --pretrained=True ...
```

When fine-tuning mobilenet v3-small based on Aircraft dataset, this tutorial mainly made the following changes to the hyper-parameters:

| Hyper-parameter | Pretrain   | Fine-tune            |
| --------------- | ---------- | -------------------- |
| dataset         | "imagenet" | ""                   |
| batch_size      | 75         | 8                    |
| image_resize    | 224        | 600                  |
| auto_augment    | -          | "randaug-m7-mstd0.5" |
| num_classes     | 1000       | 100                  |
| pretrained      | False      | True                 |
| epoch_size      | 470        | 50                   |
| lr              | 0.77       | 0.002                |

### Fine-tuning All the Parameters

Since the progress of this type of fine-tuning is the same as training from scratch, simply **start the training by running finetune.py** and adjust the parameters as training from scratch.

### Freeze Feature Network

#### Linear Probe

We prevent parameters from updating by setting `requires_grad=False` for all parameters except those in the full connection layer. In finetune.py, add the following code after `create_model` :

```python
from mindcv.models.registry import _model_pretrained_cfgs

# ...create_model()

# number of parameters to be updated
num_params = 2

# read names of parameters in FC layer
classifier_names = [_model_pretrained_cfgs[args.model]["classifier"] + ".weight",
                    _model_pretrained_cfgs[args.model]["classifier"] + ".bias"]

# prevent parameters in network(except the classifier) from updating
for param in network.trainable_params():
    if param.name not in classifier_names:
        param.requires_grad = False
```

#### Freeze Part of the Feature Network

To balance the speed and precision of fine-tuning, we can also fix some target network parameters and train the parameters in the deep layer only. It is necessary to extract the parameter names in those layers to be frozen and slightly modify the code in the last chapter. By printing the result of `create_model` -- `network`, we can see that in MindCV, each layer of the network of mobilenet v3-small is named with `features.*`. Suppose that we freeze only the first 7 layers of the network, add the following code after `create_model`:

```python
# ...create_model()

# read names of network layers
freeze_layer=["features."+str(i) for i in range(7)]

# prevent parameters in the first 7 layers of the network from updating
for param in network.trainable_params():
    for layer in freeze_layer:
        if layer in param.name:
            param.requires_grad = False
```

### Set Learning Rate for Specific Layers

To further improve the training accuracy of a fine-tuned model, we can set different learning rates for different layers in the network. This is because the shallow part of the network generally recognizes common contours or features, so even if parameters in this part will be updated, the learning rate should be set relatively small; The deep part generally recognizes the detailed personal characteristics of an object, so the learning rate can be set relatively large; Compared with the feature network that needs to retain the pre-training information as much as possible, the classifier needs to be trained from the beginning, hence the learning rate can be appropriately increased. Since this operation is elaborate, we need to enter finetune.py to specify the parameter names of specific layers and the corresponding learning rates.

MindCV uses [` create_optimizer `](https://mindspore-lab.github.io/mindcv/zh/reference/optim/#mindcv.optim.optim_factory.create_optimizer) to generate the optimizer and passes the learning rate to the optimizer. To set the tiered learning rate, simply **change the `params` parameter of `create_optimizer` function in finetune.py from `network.trainable_params()` to a list containing the names of the specific parameters and the corresponding learning rate**, which you can refer to the [API documentation of optimizers](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore.nn.html#%E4%BC%98%E5%8C%96%E5%99%A8). The specific structure of the network and the parameter names in each layer can be viewed by printing the result of `create_model` -- `network`.

> Tips: You can also use the same operation to set different weight_decay for parameters.

#### Set Learning Rate for Classifier

Taking mobilenet v3-small as an example, the model classifier name starts with "classifier", so if we only increase the learning rate of the classifier, we need to specify it at each step of training. `lr_scheduler` is a learning rate list generated by `create_scheduler`, which contains the learning rate at each step of training. Suppose we adjust the learning rate of the classifier to 1.2 times that on the feature network. The changes to the finetune.py code are as follows:

```python
# ...


# Note: a)the params-lr dict must contain all the parameters. b)Also, you're recommended to set a dict with a key "order_params" to make sure the parameters will be updated in the right order.
params_lr_group = [{"params": list(filter(lambda x: 'classifier' in x.name, network.trainable_params())),
                    "lr": [i*1.2 for i in lr_scheduler]},
                   {"params": list(filter(lambda x: 'classifier' not in x.name, network.trainable_params())),
                    "lr": lr_scheduler},
                   {"order_params": network.trainable_params()}]

optimizer = create_optimizer(params_lr_group,
                             opt=args.opt,
                             lr=lr_scheduler,
                             ...)
```

#### Set Learning Rate for Any Layers in Feature Network

Similar to adjusting the learning rate of a classifier alone, setting the learning rate of layers in feature network requires a list specifying the learning rate for each layer. Assuming that we only increase the learning rate of the last three layers of the feature network (with prefix features.13, features.14, features.15), the code for creating the optimizer in finetune.py will be changed as follows:

```python
# ...


# Note: a)the params-lr dict must contain all the parameters. b)Also, you're recommended to set a dict with a key "order_params" to make sure the parameters will be updated in the right order.
params_lr_group = [{"params": list(filter(lambda x: 'features.13' in x.name, network.trainable_params())),
                    "lr": [i * 1.05 for i in lr_scheduler]},
                   {"params": list(filter(lambda x: 'features.14' in x.name, network.trainable_params())),
                    "lr": [i * 1.1 for i in lr_scheduler]},
                   {"params": list(filter(lambda x: 'features.15' in x.name, network.trainable_params())),
                    "lr": [i * 1.15 for i in lr_scheduler]},
                   {"params": list(filter(
                       lambda x: ".".join(x.name.split(".")[:2]) not in ["features.13", "features.14", "features.15"],
                       network.trainable_params())),
                    "lr": lr_scheduler},
                   {"order_params": network.trainable_params()}]

optimizer = create_optimizer(params_lr_group,
                             opt=args.opt,
                             lr=lr_scheduler,
                             ...)
```


## Evaluation

After training, use the model weights stored in `*_best-ckpt` format in the./ckpt folder to evaluate the performance of the network on the test set. Just **run validate.py** and pass the file path of the model configuration file as well as the model weight to it:

```bash
python validate.py --config=./configs/mobilenetv3/mobilnet_v3_small_ascend.yaml --data_dir=./aircraft/data --ckpt_path=./ckpt/mobilenet_v3_small_100_best.ckpt
```

The following table summarizes the Top-1 accuracy of the fine-tuned mobilenet v3-small on the Aircraft dataset with the same training configuration but different fine-tuning skills:

| Network            | Freeze All the Feature Work | Freeze Shallow Part of Feature Network | Full Fine-tuning | Full Fine-tuning with Increasing Learning Rate of Classifier | Full Fine-tuning with Increasing Learning Rate of Deep Layers |
| ------------------ | --------------------------- | -------------------------------------- | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| mobilenet v3-small | 48.66%                      | 76.83%                                 | 88.35%           | 88.89%                                                       | 88.68%                                                       |

## Prediction

Refer to this section of the MindCV fine-tuning tutorial: [visual model reasoning results](https://mindspore-lab.github.io/mindcv/zh/tutorials/finetune/#12), or add the following code in validate.py to generate a text file ./ckpt/pred.txt that stores the true and predicted labels of the test set:

```python
# ... after model.eval()

# predited label
pred = np.argmax(model.predict(images).asnumpy(), axis=1)

# real label
images, labels = next(loader_eval.create_tuple_iterator())

# write pred.txt
prediction = np.array([pred, labels]).transpose()
np.savetxt("./ckpt/pred.txt", prediction, fmt="%s", header="pred \t real")
```

## Appendix

The following table shows the Top-1 accuracy (%) of full-model fine-tuning on the Aircraft dataset on several CNNs. For the classification accuracy that can be achieved on this dataset, see [Aircraft Leaderboard](https://fgvc.org/leaderboard/Aircraft.html) and [Paper With Code](https://paperswithcode.com/sota/fine-grained-image-classification-on-fgvc).

| Network            | Full Fine-tuning Accuracy with Mindcv | Accuracy in Papers                                           |
| ------------------ | ------------------------------------- | ------------------------------------------------------------ |
| mobilenet v3-small | 88.35%                                | -                                                            |
| mobilenet v3-large | 92.22%                                | [83.8%](https://opus.lib.uts.edu.au/handle/10453/156186)     |
| convnext-tiny      | 93.69%                                | [84.23%](http://ise.thss.tsinghua.edu.cn/~mlong/doc/hub-pathway-transfer-learning-nips22.pdf) |
| resnest50          | 86.82%                                | -                                                            |
