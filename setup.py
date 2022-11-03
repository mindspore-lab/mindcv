#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('mindcv/version.py').read())

setup(
    name="mindcv",
    author="MindSpore Ecosystem",
    author_email='mindspore-ecosystem@example.com',
    url="https://github.com/mindspore-ecosystem/mindcv",
    project_urls={
        'Sources': 'https://github.com/mindspore-ecosystem/mindcv',
        'Issue Tracker': 'https://github.com/mindspore-ecosystem/mindcv/issues',
    },
    description="A toolbox of vision models and algorithms based on MindSpore.",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=['mindcv', 'mindcv.*']),
    install_requires=[
        'numpy >= 1.17.0',
        'PyYAML >= 5.3',
        'tqdm'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    tests_require=[
        'pytest',
    ],
    version=__version__,
    zip_safe=False
)
