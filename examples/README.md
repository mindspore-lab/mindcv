This folder contains examples for various tasks, which users can run easily.  

### Finetune 
```
python examples/finetune.py

```
This example shows how to finetune a pretrained model on your own dataset. You can also specifiy `--freeze_backbone` to choose whether to freeze the backbone and finetune the classifier head only.


### PyNative mode with flexible training 
```
python examples/train_cifiar_dynamic.py 
```
This example shows how to use pynative mode and apply a flexible training strategy for easy debug. If you want to change back to graph mode to save training time, run

```
python examples/train_cifiar_dynamic.py --dynamic=0 --train_step=0 
```
