# Utility Scripts

This folder is a collection of utility scripts, listed and explained below.

> All scripts need to be run in the root path of project, unless otherwise noted.

## gen_benchmark.py

Generating benchmark by collecting results from [configs](../configs) folder. Usage:

```shell
python ./scripts/gen_benchmark.py
```

It will generate a markdown file, named as `benchmark_results.md`.

## package.sh(Deprecated)

Making wheel package of `mindcv` and sha256sum of the wheel files. Usage:

```shell
./scripts/package.sh
```

**New**! Just simply run the following command to make the wheel:

```shell
python -m build
```

## launch_dist.sh or launch_dist.py

A simple clean launcher for distributed training on **_Ascend_**.
Following [instruction](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/parallel/startup_method.html) from Mindspore,
except launching distributed training with `mpirun`, we can also use multiprocess
with multi-card networking configuration `rank_table.json` to manually start a process on each card.
To get `rank_table.json` on your machine, try the hccl tools from [here](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

> After you get the `rank_table.json`, replace the `"/path/to/rank_table.json"` in `launch_dist.sh` with the actual path.

Now, you can replace your standalone launching with distributed launching:

```diff
- python script.py --arg1=value1 --arg2=value2
+ ./scripts/launch_dist.sh script.py --arg1=value1 --arg2=value2
```

where `--arg*` are arguments of `script.py`.

For example:

```shell
./scripts/launch_dist.sh train.py --config=configs/resnet/resnet_50_ascend.yaml --data_dir=/my/awesome/dataset
```

> Note: Don't forget to check the argument `--distribute` if you are using `train.py` or `train_with_func.py`!

For anyone who hates shell scripts, we offer python scripts `launch_dist.py` as well. Both are used in the same way!
