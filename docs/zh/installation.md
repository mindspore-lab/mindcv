---
hide:
  - navigation
---

## 依赖

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (分布式训练所需)

为了安装`python`相关库依赖，只需运行：

```shell
pip install -r requirements.txt
```

!!! tip

    [MindSpore]可以通过遵循[官方指引](https://www.mindspore.cn/install)，在不同的硬件平台上获得最优的安装体验。
    为了在分布式模式下运行，您还需要安装[OpenMPI]。

如下的指引假设您已经完成了所有依赖库的安装。

## PyPI源安装

MindCV发布为一个[Python包]并能够通过`pip`进行安装。我们推荐您在[虚拟环境]安装使用。 打开终端，输入以下指令来安装MindCV:

=== "stable"

    ``` shell
    pip install mindcv
    ```

=== "nightly"

    ``` shell
    # 暂不支持
    ```

上述命令会自动安装依赖：[NumPy]，[PyYAML] 和 [tqdm]的兼容版本。

!!! tip

    如果您之前没有使用 Python 的经验，我们建议您阅读[使用Python的pip来管理您的项目的依赖关系]，
    这是对 Python 包管理机制的一个很好的介绍，并且可以帮助您在遇到错误时进行故障排除。

!!! warning

    上述命令 **不会** 安装[MindSpore].
    我们强烈推荐您通过[官方指引](https://www.mindspore.cn/install)来安装[MindSpore]。

[Python包]: https://pypi.org/project/mindcv/
[虚拟环境]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
[MindSpore]: https://www.mindspore.cn/
[OpenMPI]: https://www.open-mpi.org/
[NumPy]: https://numpy.org/
[PyYAML]: https://pyyaml.org/
[tqdm]: https://tqdm.github.io/
[使用Python的pip来管理您的项目的依赖关系]: https://realpython.com/what-is-pip/


## 源码安装 (未经测试版本)

### from VSC

```shell
pip install git+https://github.com/mindspore-lab/mindcv.git
```

### from local src

!!! tip

    由于本项目处于活跃开发阶段，如果您是开发者或者贡献者，请优先选择此安装方式。

MindCV可以在由 [GitHub] 克隆仓库到本地文件夹后直接使用。 这对于想使用最新版本的开发者十分方便:

```shell
git clone https://github.com/mindspore-lab/mindcv.git
```

在克隆到本地之后，推荐您使用"可编辑"模式进行安装，这有助于解决潜在的模块导入问题。

```shell
cd mindcv
pip install -e .
```

[GitHub]: https://github.com/mindspore-lab/mindcv
