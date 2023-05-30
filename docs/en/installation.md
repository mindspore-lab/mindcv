---
hide:
  - navigation
---

## Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (for distributed mode)

To install the python library dependency, just run:

```shell
pip install -r requirements.txt
```

!!! tip

    [MindSpore] can be easily installed by following the [official instructions](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit.
    To run in distributed mode, [OpenMPI] is required to install.

The following instructions assume the desired dependency is fulfilled.

## Install with PyPI

MindCV is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. Open up a terminal and install
MindCV with:

=== "stable"

    ``` shell
    pip install mindcv
    ```

=== "nightly"

    ``` shell
    # working on it using test.pypi
    ```

This will automatically install compatible versions of dependencies:
[NumPy], [PyYAML] and [tqdm].

!!! tip

    If you don't have prior experience with Python, we recommend reading
    [Using Python's pip to Manage Your Projects' Dependencies], which is a really
    good introduction on the mechanics of Python package management and helps you
    troubleshoot if you run into errors.

!!! warning

    The above command will **NOT** install [MindSpore].
    We highly recommand you install [MindSpore] following the [official instructions](https://www.mindspore.cn/install).

[Python package]: https://pypi.org/project/mindcv/
[virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
[MindSpore]: https://www.mindspore.cn/
[OpenMPI]: https://www.open-mpi.org/
[NumPy]: https://numpy.org/
[PyYAML]: https://pyyaml.org/
[tqdm]: https://tqdm.github.io/
[Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/


## Install from Source (Bleeding Edge Version)

### from VCS

```shell
pip install git+https://github.com/mindspore-lab/mindcv.git
```

### from local src

!!! tip

    As this project is in active development, if you are a developer or contributor, please prefer this installation!

MindCV can be directly used from [GitHub] by cloning the repository into a local folder which might be useful if you want to use the very latest version:

```shell
git clone https://github.com/mindspore-lab/mindcv.git
```

After cloning from `git`, it is recommended that you install using "editable" mode, which can help resolve potential module import issues:

```shell
cd mindcv
pip install -e .
```

[GitHub]: https://github.com/mindspore-lab/mindcv
