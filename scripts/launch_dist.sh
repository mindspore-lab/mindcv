#!/bin/bash
# Usage:
# ./scripts/launch_dist.sh script.py --arg1=value1 --arg2=value2
# Example:
# ./scripts/launch_dist.sh train.py --config=configs/resnet/resnet_50_ascend.yaml --data_dir=/my/awesome/dataset

export RANK_SIZE=8
export RANK_TABLE_FILE="/path/to/rank_table.json"


echo "Script: $1, Args: ${@:2}"  # ${parameter:offset:length}

# trap SIGINT to execute kill 0, which will kill all processes
trap 'kill 0' SIGINT
for ((i = 0; i < ${RANK_SIZE}; i++)); do
    export RANK_ID=$i
    export DEVICE_ID=$i
    echo "Launching rank: ${RANK_ID}, device: ${DEVICE_ID}"
    python -u $@ &
done
# wait for all processes to finish
wait
