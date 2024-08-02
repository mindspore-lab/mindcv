# MindCV 贡献指南

欢迎贡献，我们将不胜感激！每一份贡献都是有益的，请接受我们的赞扬。

## 贡献类型

### 报告错误

报告错误至 https://github.com/mindspore-lab/mindcv/issues.

如果您要报告错误，请包括：

* 您的操作系统名称和版本。
* 任何可能有助于故障排除的本地设置详细信息。
* 重现错误的详细步骤。

### 修复Bugs

查阅GitHub issues以了解Bugs。任何带有“bug”和“help wanted”标签的issue都对想要解决它的人开放。

### 实现features

查阅GitHub issues以了解features。任何标有“enhancement”和“help wanted”的issue都对想要实现它的人开放。

### 编写文档

MindCV通常可以使用多种方式编写文档，可以编写在官方MindCV文档中，或者编写在docstrings中，甚至可以编写在网络上的博客、文章上。

### 提交反馈

发送反馈的最佳方式是在 https://github.com/mindspore-lab/mindcv/issues 上提交问题。

如果您要提出一项功能：

* 详细说明它将如何工作。
* 尽可能缩小范围，使其更易于实施。
* 请记住，这是一个志愿者驱动的项目，欢迎贡献 :)

## 入门

准备好贡献了吗？以下是如何设置 `mindcv` 进行本地开发。

1. 在 [GitHub](https://github.com/mindlab-ai/mindcv) 上 fork `mindcv` 代码仓。
2. 在本地克隆您的 fork：

```shell
git clone git@github.com:your_name_here/mindcv.git
```

之后，您应该将官方代码仓添加为upstream代码仓：

```shell
git remote add upper git@github.com:mindspore-lab/mindcv
```

3. 将本地副本配置到 conda 环境中。假设您已安装 conda，您可以按照以下方式设置 fork 以进行本地开发：

```shell
conda create -n mindcv python=3.8
conda activate mindcv
cd mindcv
pip install -e 。
```

4. 为本地开发创建一个分支：

```shell
git checkout -b name-of-your-bugfix-or-feature
```

现在您可以在本地进行更改。

5. 完成更改后，检查您的更改是否通过了linters和tests检查：

```shell
pre-commit run --show-diff-on-failure --color=always --all-files
pytest
```

如果所有静态 linting 都通过，您将获得如下输出：

![pre-commit-succeed](https://user-images.githubusercontent.com/74176172/221346245-ea868015-bb09-4e53-aa56-73b015e1e336.png)

否则，您需要根据输出修复警告：

![pre-commit-failed](https://user-images.githubusercontent.com/74176172/221346251-7d8f531f-9094-474b-97f0-fd5a55e6d3de.png)

要获取 pre-commit 和 pytest，只需使用 pip 安装它们到您的 conda 环境中即可。

6. 提交您的更改并将您的分支推送到 GitHub：

```shell
git add .
git commit -m “您对更改的详细描述。”
git push origin name-of-your-bugfix-or-feature
```

7. 通过 GitHub 网站提交pull request。

## pull request指南

在提交pull request之前，请检查它是否符合以下指南：

1. pull request应包括测试。
2. 如果pull request添加了功能，则应更新文档。将新功能放入带有docstring的函数中，并将特性添加到 README.md 中的列表中。
3. pull request应适用于 Python 3.7、3.8 和 3.9 以及 PyPy。检查
   https://github.com/mindspore-lab/mindcv/actions
   并确保所有受支持的 Python 版本的测试都通过。

## 提示

您可以安装 git hook脚本，而不是手动使用 `pre-commit run -a` 进行 linting。

运行flowing command来设置 git hook脚本

```shell
pre-commit install
```

现在 `pre-commit` 将在 `git commit` 上自动运行！

## 发布

提醒维护者如何部署。确保提交所有更改（包括 HISTORY.md 中的条目），然后运行：

```shell
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

如果测试通过，GitHub Action 将部署到 PyPI。
