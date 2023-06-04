#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# read the `__version__` global variable in `version.py`
exec(open("mindcv/version.py").read())

setup(
    name="mindcv",
    author="MindSpore Lab",
    author_email="mindspore-lab@example.com",
    url="https://github.com/mindspore-lab/mindcv",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindcv",
        "Issue Tracker": "https://github.com/mindspore-lab/mindcv/issues",
    },
    description="A toolbox of vision models and algorithms based on MindSpore.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["mindcv", "mindcv.*"]),
    install_requires=[
        "numpy >= 1.17.0",
        "PyYAML >= 5.3",
        "tqdm",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    tests_require=[
        "pytest",
    ],
    version=__version__,
    zip_safe=False,
)
