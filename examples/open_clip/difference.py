"""
With 2 folder paths provided,
this script will output absolute as well as relative difference of paired files.

Examples:
python difference.py --a_path=./results_a/ --b_path=./results_b/

"""

import argparse
import os
import sys

import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a_path",
        type=str,
        default=None,
        help="A folder path containing at least one .txt file of model results",
    )
    parser.add_argument(
        "--b_path",
        type=str,
        default=None,
        help="Another folder path containing at least one .txt file of model results",
    )
    args = parser.parse_args(args)
    return args


def difference(a_file, b_file):
    file = open(a_file, "r")
    a_variable = eval(file.read())
    file.close()

    file = open(b_file, "r")
    b_variable = eval(file.read())
    file.close()

    if not isinstance(b_variable, np.ndarray):
        b_variable = np.array(b_variable)
    if not isinstance(a_variable, np.ndarray):
        a_variable = np.array(a_variable)

    if b_variable.shape != a_variable.shape:
        raise ValueError(
            f"{a_variable} has shape {a_variable.shape} "
            f"while {b_variable} has different shape of {b_variable.shape}."
        )

    # abs diff (mean)
    abs_mean = abs(a_variable - b_variable).mean()

    # abs diff (max)
    abs_max = abs(a_variable - b_variable).max()

    # relative diff (mean)
    rel_mean = (abs(a_variable - b_variable) / (abs(b_variable) + 1e-6)).mean()

    # relative diff (max)
    rel_max = (abs(a_variable - b_variable) / (abs(b_variable) + 1e-6)).max()

    print(
        f'{os.path.basename(a_file).replace(".txt",": ")}\n abs_mean: {abs_mean}\n '
        f"abs_max: {abs_max}\n rel_mean: {rel_mean}\n rel_max: {rel_max}\n\n"
    )


def main(args):
    args = parse_args(args)

    a_files = []
    b_files = []
    for root, dirs, files in os.walk(args.a_path):
        for file in files:
            a_files.append(os.path.join(root, file))
    a_files = sorted(a_files)
    for root, dirs, files in os.walk(args.b_path):
        for file in files:
            b_files.append(os.path.join(root, file))
    b_files = sorted(b_files)

    if len(a_files) != len(b_files):
        raise ValueError(f"Files in {args.a_path} are diiferent with those in {args.b_path}.")
    for file in range(len(a_files)):
        if os.path.basename(a_files[file]) != os.path.basename(b_files[file]):
            raise ValueError(f"Files in {args.a_path} are diiferent with those in {args.b_path}.")
        difference(a_files[file], b_files[file])


if __name__ == "__main__":
    main(sys.argv[1:])
