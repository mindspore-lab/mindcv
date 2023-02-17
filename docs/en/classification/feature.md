# Features

- **Easy-to-Use.** MindCV decomposes the vision framework into various configurable components. It is easy to customize your data pipeline, models, and learning pipeline with MindCV:

```python
>>> import mindcv
# create a dataset
>>> dataset = mindcv.create_dataset('cifar10', download=True)
# create a model
>>> network = mindcv.create_model('resnet50', pretrained=True)
```

Users can customize and launch their transfer learning or training task in one command line.

``` python
# transfer learning in one command line
>>> !python train.py --model=swin_tiny --pretrained --opt=adamw --lr=0.001 --data_dir={data_dir}
```

- **State-of-The-Art.** MindCV provides various CNN-based and Transformer-based vision models including SwinTransformer. Their pretrained weights and performance reports are provided to help users select and reuse the right model.

- **Flexibility and efficiency.** MindCV is bulit on MindSpore which is an efficent DL framework that can run on different hardward platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibity.
