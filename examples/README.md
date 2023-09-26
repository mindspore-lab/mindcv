This folder contains examples for various tasks, which users can run easily.

### Single process with model training and evaluation
```
python examples/train_with_func_example.py
```
This example shows how to train and evaluate a model on your own dataset.

### Multiprocess with model training and evaluation
```
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python examples/train_parallel_with_func_example.py
```
This example shows how to train and evaluate a mode with multiprocess on your own dataset on GPU.
