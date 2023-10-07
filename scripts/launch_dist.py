#!/usr/bin/env python
# Usage:
# ./scripts/launch_dist.py script.py --arg1=value1 --arg2=value2
# Example:
# ./scripts/launch_dist.py train.py --config=configs/resnet/resnet_50_ascend.yaml --data_dir=/my/awesome/dataset

import multiprocessing as mp
import os
import sys

BIAS = 0
RANK_SIZE = 8
RANK_TABLE_FILE = "/path/to/rank_table.json"


def worker(rank_id, script, args):
    os.environ["RANK_ID"] = f"{rank_id}"  # logical id
    os.environ["DEVICE_ID"] = f"{rank_id + BIAS}"  # physical id
    os.environ["RANK_TABLE_FILE"] = RANK_TABLE_FILE
    print(f"Launching rank: {os.getenv('RANK_ID')}, device: {os.getenv('DEVICE_ID')}, pid: {os.getpid()}")
    os.system(f"python -u {script} {args}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    script_, args_ = sys.argv[1], " ".join(sys.argv[2:])
    print(f"Script: {script_}, Args: {args_}")
    processes = [mp.Process(target=worker, args=(i, script_, args_)) for i in range(RANK_SIZE)]
    [p.start() for p in processes]
    [p.join() for p in processes]
