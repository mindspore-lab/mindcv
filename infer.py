"""MindSpore Inference Script

Example:
    $ python inference.py --image-path="/path/to/image.png" --model="densenet121"
"""

import ast
import argparse
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn

from mindcv.models import create_model
from mindcv.data import create_transforms

parser = argparse.ArgumentParser(description='MindSpore Inference Demo')
parser.add_argument('--image_path', type=str, help='path to image')
parser.add_argument('--model', type=str, help='name of model')
parser.add_argument('--ckpt_path', type=str, help='checkpoint path')


def main():
    args = parser.parse_args()
    ms.set_seed(1)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    img = Image.open(args.image_path).convert("RGB")
    # create transform
    transform_list = create_transforms(
        dataset_name="imagenet",
        is_training=False
    )
    transform_list.pop(0)
    for transform in transform_list:
        img = transform(img)
    img = np.expand_dims(img, axis=0)

    # create model
    network = create_model(
        model_name=args.model,
        pretrained=True
    )
    network.set_train(False)
    logits = nn.Softmax()(network(ms.Tensor(img)))[0].asnumpy()
    preds = np.argsort(logits)[::-1][:5]
    probs = logits[preds]
    with open("./tutorials/imagenet1000_clsidx_to_labels.txt", encoding='utf-8') as f:
        idx2label = ast.literal_eval(f.read())
    #print(f"Predict result of {args.image_path}:")
    cls_prob = {}
    for pred, prob in zip(preds, probs):
        cls_name = idx2label[pred]
        cls_prob[cls_name] = prob
    print(cls_prob)
        


if __name__ == '__main__':
    main()
