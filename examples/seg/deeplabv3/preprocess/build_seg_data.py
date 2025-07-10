# Copy from https://github.com/mindspore-ai/models/blob/master/official/cv/DeepLabv3/src/data/build_seg_data.py

import argparse
import os

from mindspore.mindrecord import FileWriter

seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}


def parse_args():
    parser = argparse.ArgumentParser("mindrecord")

    parser.add_argument("--data_root", type=str, default="", help="root path of data")
    parser.add_argument("--data_list", type=str, default="", help="list of data")
    parser.add_argument("--dst_path", type=str, default="", help="save path of mindrecords")
    parser.add_argument("--num_shards", type=int, default=8, help="number of shards")

    parser_args, _ = parser.parse_known_args()
    return parser_args


if __name__ == "__main__":
    args = parse_args()

    data = []
    with open(args.data_list) as f:
        lines = f.readlines()

    dst_dir = "/".join(args.dst_path.split("/")[:-1])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print("number of samples:", len(lines))
    writer = FileWriter(file_name=args.dst_path, shard_num=args.num_shards)
    writer.add_schema(seg_schema, "seg_schema")
    cnt = 0
    for line in lines:
        img_path, label_path = line.strip().split(" ")
        sample_ = {"file_name": img_path.split("/")[-1]}
        with open(os.path.join(args.data_root, img_path), "rb") as f:
            sample_["data"] = f.read()
        with open(os.path.join(args.data_root, label_path), "rb") as f:
            sample_["label"] = f.read()
        data.append(sample_)
        cnt += 1
        if cnt % 1000 == 0:
            writer.write_raw_data(data)
            print("number of samples written:", cnt)
            data = []

    if data:
        writer.write_raw_data(data)
    writer.commit()
    print("number of samples written:", cnt)
