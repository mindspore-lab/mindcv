"""
With 2 folder paths provided,
this script will output absolute as well as relative difference of paired files.

Examples:
python difference.py --ms_path=./ms_data/ --to_path=./ms_data/

"""

import argparse
import os
import sys

import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_path",
        type=str,
        default=None,
        help="A folder path containing at least one .txt file of Mindspore's results",
    )
    parser.add_argument(
        "--torch_path",
        type=str,
        default=None,
        help="A folder path containing at least one .txt file of PyTorch's results",
    )
    args = parser.parse_args(args)
    return args


def difference(ms_file, torch_file):
    file = open(ms_file, "r")
    ms_variable = eval(file.read())
    file.close()

    file = open(torch_file, "r")
    torch_variable = eval(file.read())
    file.close()

    if not isinstance(torch_variable, np.ndarray):
        torch_variable = np.array(torch_variable)
    if not isinstance(ms_variable, np.ndarray):
        ms_variable = np.array(ms_variable)

    if torch_variable.shape != ms_variable.shape:
        raise ValueError(
            f"{ms_variable} has shape {ms_variable.shape} "
            f"while {torch_variable} has different shape of {torch_variable.shape}."
        )

    # abs diff (mean)
    abs_mean = abs(ms_variable - torch_variable).mean()

    # abs diff (max)
    abs_max = abs(ms_variable - torch_variable).max()

    # relative diff (mean)
    rel_mean = (abs(ms_variable - torch_variable) / (abs(torch_variable) + 1e-6)).mean()

    # relative diff (max)
    rel_max = (abs(ms_variable - torch_variable) / (abs(torch_variable) + 1e-6)).max()

    print(
        f'{os.path.basename(ms_file).replace(".txt",": ")}\n abs_mean: {abs_mean}\n '
        f"abs_max: {abs_max}\n rel_mean: {rel_mean}\n rel_max: {rel_max}\n\n"
    )


def main(args):
    args = parse_args(args)

    ms_files = []
    torch_files = []
    for root, dirs, files in os.walk(args.ms_path):
        for file in files:
            ms_files.append(os.path.join(root, file))
    ms_files = sorted(ms_files)
    for root, dirs, files in os.walk(args.torch_path):
        for file in files:
            torch_files.append(os.path.join(root, file))
    torch_files = sorted(torch_files)

    if len(ms_files) != len(torch_files):
        raise ValueError(f"Files in {args.ms_path} are diiferent with those in {args.torch_path}.")
    for file in range(len(ms_files)):
        if os.path.basename(ms_files[file]) != os.path.basename(torch_files[file]):
            raise ValueError(f"Files in {args.ms_path} are diiferent with those in {args.torch_path}.")
        difference(ms_files[file], torch_files[file])


if __name__ == "__main__":
    main(sys.argv[1:])
