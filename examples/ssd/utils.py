import itertools as it
import json
import math

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mindspore.nn as nn
from mindspore import Tensor


class GridAnchorGenerator:
    """
    Anchor Generator
    """

    def __init__(self, image_shape, scale, scales_per_octave, aspect_ratios):
        super(GridAnchorGenerator, self).__init__()
        self.scale = scale
        self.scales_per_octave = scales_per_octave
        self.aspect_ratios = aspect_ratios
        self.image_shape = image_shape

    def generate(self, step):
        scales = np.array(
            [2 ** (float(scale) / self.scales_per_octave) for scale in range(self.scales_per_octave)]
        ).astype(np.float32)
        aspects = np.array(list(self.aspect_ratios)).astype(np.float32)

        scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspects)
        scales_grid = scales_grid.reshape([-1])
        aspect_ratios_grid = aspect_ratios_grid.reshape([-1])

        feature_size = [self.image_shape[0] / step, self.image_shape[1] / step]
        grid_height, grid_width = feature_size

        base_size = np.array([self.scale * step, self.scale * step]).astype(np.float32)
        anchor_offset = step / 2.0

        ratio_sqrt = np.sqrt(aspect_ratios_grid)
        heights = scales_grid / ratio_sqrt * base_size[0]
        widths = scales_grid * ratio_sqrt * base_size[1]

        y_centers = np.arange(grid_height).astype(np.float32)
        y_centers = y_centers * step + anchor_offset
        x_centers = np.arange(grid_width).astype(np.float32)
        x_centers = x_centers * step + anchor_offset
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        x_centers_shape = x_centers.shape
        y_centers_shape = y_centers.shape

        widths_grid, x_centers_grid = np.meshgrid(widths, x_centers.reshape([-1]))
        heights_grid, y_centers_grid = np.meshgrid(heights, y_centers.reshape([-1]))

        x_centers_grid = x_centers_grid.reshape(*x_centers_shape, -1)
        y_centers_grid = y_centers_grid.reshape(*y_centers_shape, -1)
        widths_grid = widths_grid.reshape(-1, *x_centers_shape)
        heights_grid = heights_grid.reshape(-1, *y_centers_shape)

        bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=3)
        bbox_sizes = np.stack([heights_grid, widths_grid], axis=3)
        bbox_centers = bbox_centers.reshape([-1, 2])
        bbox_sizes = bbox_sizes.reshape([-1, 2])
        bbox_corners = np.concatenate([bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes], axis=1)
        self.bbox_corners = bbox_corners / np.array([*self.image_shape, *self.image_shape]).astype(np.float32)
        self.bbox_centers = np.concatenate([bbox_centers, bbox_sizes], axis=1)
        self.bbox_centers = self.bbox_centers / np.array([*self.image_shape, *self.image_shape]).astype(np.float32)

        return self.bbox_centers, self.bbox_corners

    def generate_multi_levels(self, steps):
        bbox_centers_list = []
        bbox_corners_list = []

        for step in steps:
            bbox_centers, bbox_corners = self.generate(step)
            bbox_centers_list.append(bbox_centers)
            bbox_corners_list.append(bbox_corners)

        self.bbox_centers = np.concatenate(bbox_centers_list, axis=0)
        self.bbox_corners = np.concatenate(bbox_corners_list, axis=0)

        return self.bbox_centers, self.bbox_corners


class GeneratDefaultBoxes:
    """
    Generate Default boxes for SSD, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_tlbr` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """

    def __init__(self, args):
        fk = args.image_size[0] / np.array(args.steps)
        scale_rate = (args.max_scale - args.min_scale) / (len(args.num_default) - 1)
        scales = [args.min_scale + scale_rate * i for i in range(len(args.num_default))] + [1.0]
        self.default_boxes = []

        for idex, feature_size in enumerate(args.feature_size):
            sk1 = scales[idex]
            sk2 = scales[idex + 1]
            sk3 = math.sqrt(sk1 * sk2)

            if idex == 0 and not args.aspect_ratios[idex]:
                w, h = sk1 * math.sqrt(2), sk1 / math.sqrt(2)
                all_sizes = [(0.1, 0.1), (w, h), (h, w)]
            else:
                all_sizes = [(sk1, sk1)]

                for aspect_ratio in args.aspect_ratios[idex]:
                    w, h = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                    all_sizes.append((w, h))
                    all_sizes.append((h, w))

                all_sizes.append((sk3, sk3))

            assert len(all_sizes) == args.num_default[idex]

            for i, j in it.product(range(feature_size), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    self.default_boxes.append([cy, cx, h, w])

        def to_tlbr(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_tlbr = np.array(tuple(to_tlbr(*i) for i in self.default_boxes), dtype="float32")
        self.default_boxes = np.array(self.default_boxes, dtype="float32")


def ssd_bboxes_encode(boxes, args):
    """
    Labels anchors with ground truth inputs.

    Args:
        boxex: ground truth with shape [N, 5], for each row, it stores [y, x, h, w, cls].

    Returns:
        gt_loc: location ground truth with shape [num_anchors, 4].
        gt_label: class ground truth with shape [num_anchors, 1].
        num_matched_boxes: number of positives in an image.
    """
    if hasattr(args, "use_anchor_generator") and args.use_anchor_generator:
        generator = GridAnchorGenerator(args.image_size, 4, 2, [1.0, 2.0, 0.5])
        default_boxes, default_boxes_tlbr = generator.generate_multi_levels(args.steps)
    else:
        generator = GeneratDefaultBoxes(args)
        default_boxes_tlbr = generator.default_boxes_tlbr
        default_boxes = generator.default_boxes

    y1, x1, y2, x2 = np.split(default_boxes_tlbr[:, :4], 4, axis=-1)
    vol_anchors = (x2 - x1) * (y2 - y1)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bbox and volume.
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.0)
        h = np.maximum(ymax - ymin, 0.0)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros((args.num_ssd_boxes), dtype=np.float32)
    t_boxes = np.zeros((args.num_ssd_boxes, 4), dtype=np.float32)
    t_label = np.zeros((args.num_ssd_boxes), dtype=np.int64)

    for bbox in boxes:
        label = int(bbox[4])
        scores = jaccard_with_anchors(bbox)
        idx = np.argmax(scores)
        scores[idx] = 2.0
        mask = scores > args.match_threshold
        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores * mask)
        t_label = mask * label + (1 - mask) * t_label

        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)

    # Transform to tlbr.
    bboxes = np.zeros((args.num_ssd_boxes, 4), dtype=np.float32)
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

    # Encode features.
    bboxes_t = bboxes[index]
    default_boxes_t = default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / (default_boxes_t[:, 2:] * args.prior_scaling[0])
    tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
    bboxes_t[:, 2:4] = np.log(tmp) / args.prior_scaling[1]
    bboxes[index] = bboxes_t

    num_match = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    return bboxes, t_label.astype(np.int32), num_match


def ssd_bboxes_decode(boxes, args):
    """Decode predict boxes to [y, x, h, w]"""
    if hasattr(args, "use_anchor_generator") and args.use_anchor_generator:
        generator = GridAnchorGenerator(args.image_size, 4, 2, [1.0, 2.0, 0.5])
        default_boxes, _ = generator.generate_multi_levels(args.steps)
    else:
        default_boxes = GeneratDefaultBoxes(args).default_boxes

    boxes_t = boxes.copy()
    # default_boxes_t = default_boxes.copy()
    boxes_t[:, :2] = boxes_t[:, :2] * args.prior_scaling[0] * default_boxes[:, 2:] + default_boxes[:, :2]
    boxes_t[:, 2:4] = np.exp(boxes_t[:, 2:4] * args.prior_scaling[1]) * default_boxes[:, 2:4]

    bboxes = np.zeros((len(boxes_t), 4), dtype=np.float32)

    bboxes[:, [0, 1]] = boxes_t[:, [0, 1]] - boxes_t[:, [2, 3]] / 2
    bboxes[:, [2, 3]] = boxes_t[:, [0, 1]] + boxes_t[:, [2, 3]] / 2

    return np.clip(bboxes, 0, 1)


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


def get_ssd_lr_scheduler(args, steps_per_epoch):
    """
    generate learning rate array for training
    """
    lr_init = args.lr_init
    lr_end = args.lr_end_rate * args.lr
    lr_max = args.lr
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epoch_size

    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = (
                lr_end
                + (lr_max - lr_end)
                * (1.0 + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps)))
                / 2.0
            )

        if lr < 0.0:
            lr = 0.0

        lr_each_step.append(lr)

    learning_rate = np.array(lr_each_step).astype(np.float32)

    return learning_rate


def get_ssd_optimizer(model, lr, args):
    optimizer = nn.Momentum(
        filter(lambda x: x.requires_grad, model.get_parameters()), lr, args.momentum, args.weight_decay, args.loss_scale
    )
    return optimizer


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]

    return keep


class COCOMetrics:
    """Calculate mAP of predicted bboxes."""

    def __init__(self, anno_json, classes, num_classes, min_score, nms_threshold, max_boxes):
        self.num_classes = num_classes
        self.classes = classes
        self.min_score = min_score
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes

        self.val_cls_dict = {i: cls for i, cls in enumerate(classes)}
        self.coco_gt = COCO(anno_json)
        cat_ids = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        self.class_dict = {cat["name"]: cat["id"] for cat in cat_ids}

        self.predictions = []
        self.img_ids = []

    def update(self, batch):
        pred_boxes = batch["boxes"]
        box_scores = batch["box_scores"]
        img_id = batch["img_id"]
        h, w = batch["image_shape"]

        final_boxes = []
        final_label = []
        final_score = []
        self.img_ids.append(img_id)

        for c in range(1, self.num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > self.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, self.nms_threshold, self.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [self.class_dict[self.val_cls_dict[c]]] * len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res["image_id"] = img_id
            res["bbox"] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
            res["score"] = score
            res["category_id"] = label
            self.predictions.append(res)

    def get_metrics(self):
        with open("predictions.json", "w") as f:
            json.dump(self.predictions, f)

        coco_dt = self.coco_gt.loadRes("predictions.json")
        E = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        E.params.imgIds = self.img_ids
        E.evaluate()
        E.accumulate()
        E.summarize()
        return E.stats[0]


def apply_eval(eval_param_dict):
    net = eval_param_dict["net"]
    net.set_train(False)
    ds = eval_param_dict["dataset"]
    anno_json = eval_param_dict["anno_json"]
    args = eval_param_dict["args"]
    coco_metrics = COCOMetrics(
        anno_json=anno_json,
        classes=args.classes,
        num_classes=args.num_classes,
        max_boxes=args.max_boxes,
        nms_threshold=args.nms_threshold,
        min_score=args.min_score,
    )

    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_id = data["img_id"]
        img_np = data["image"]
        image_shape = data["image_shape"]

        output = net(Tensor(img_np))

        for batch_idx in range(img_np.shape[0]):
            pred_batch = {
                "boxes": output[0].asnumpy()[batch_idx],
                "box_scores": output[1].asnumpy()[batch_idx],
                "img_id": int(np.squeeze(img_id[batch_idx])),
                "image_shape": image_shape[batch_idx],
            }
            coco_metrics.update(pred_batch)

    eval_metrics = coco_metrics.get_metrics()

    return eval_metrics
