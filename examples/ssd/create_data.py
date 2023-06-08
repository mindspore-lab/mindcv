import argparse
import os

import numpy as np

from mindspore.mindrecord import FileWriter

coco_classes = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def create_coco_label(data_path, is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = data_path

    if is_training:
        data_type = "train2017"
    else:
        data_type = "val2017"

    # Classes need to train or test.
    train_cls = coco_classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, f"annotations/instances_{data_type}.json")

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    images = []
    image_path_dict = {}
    image_anno_dict = {}
    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        iscrowd = False
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            iscrowd = iscrowd or label["iscrowd"]
            if class_name in train_cls:
                x_min, x_max = bbox[0], bbox[0] + bbox[2]
                y_min, y_max = bbox[1], bbox[1] + bbox[3]
                annos.append(list(map(round, [y_min, x_min, y_max, x_max])) + [train_cls_dict[class_name]])

        if not is_training and iscrowd:
            continue
        if len(annos) >= 1:
            images.append(img_id)
            image_path_dict[img_id] = image_path
            image_anno_dict[img_id] = np.array(annos)

    return images, image_path_dict, image_anno_dict


def data_to_mindrecord_byte_image(dataset="coco", data_path="", out_path="", is_training=True, file_num=8):
    """Create MindRecord file."""
    if is_training:
        os.mkdir(os.path.join(out_path, "train"))
        mindrecord_path = os.path.join(out_path, "train", dataset)
    else:
        os.mkdir(os.path.join(out_path, "val"))
        mindrecord_path = os.path.join(out_path, "val", dataset)

    writer = FileWriter(mindrecord_path, file_num)

    if dataset == "coco":
        images, image_path_dict, image_anno_dict = create_coco_label(data_path, is_training)
    else:
        raise NotImplementedError

    ssd_json = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ssd_json, "ssd_json")

    for img_id in images:
        image_path = image_path_dict[img_id]

        with open(image_path, "rb") as f:
            img = f.read()

        annos = np.array(image_anno_dict[img_id], dtype=np.int32)
        img_id = np.array([img_id], dtype=np.int32)
        row = {"img_id": img_id, "image": img, "annotation": annos}
        writer.write_raw_data([row])

    writer.commit()


def convert_dataset(dataset="coco", data_path="", out_path=""):
    if dataset == "coco":
        if os.path.isdir(data_path):
            print("Start converting training dataset...")
            data_to_mindrecord_byte_image(dataset=dataset, data_path=data_path, out_path=out_path, is_training=True)
            print("Training dataset conversion done.")
            print("Start converting evaluation dataset...")
            data_to_mindrecord_byte_image(dataset=dataset, data_path=data_path, out_path=out_path, is_training=False)
            print("Evaluation dataset conversion done.")
        else:
            print("data path not exits.")
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="coco", help="name of the dataset")
parser.add_argument("--data_path", type=str, default="./data/coco/", help="specify the root path of dataset")
parser.add_argument(
    "--out_path", type=str, default="./data/coco/", required=False, help="specify the path of the coverted dataset"
)
args = parser.parse_args()


if __name__ == "__main__":
    convert_dataset(dataset=args.dataset, data_path=args.data_path, out_path=args.out_path)
