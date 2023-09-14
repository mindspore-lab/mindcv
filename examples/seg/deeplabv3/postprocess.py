import os

import cv2
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor


def check_batch_size(num_samples, ori_batch_size=32, refine=True):
    if num_samples % ori_batch_size == 0:
        return ori_batch_size
    else:
        # search a batch size that is divisible by num samples.
        for bs in range(ori_batch_size - 1, 0, -1):
            if num_samples % bs == 0:
                print(
                    f"WARNING: num eval samples {num_samples} can not be divided by "
                    f"the input batch size {ori_batch_size}. The batch size is refined to {bs}"
                )
                return bs
    return 1


def calculate_hist(flattened_label, flattened_pred, n):
    k = (flattened_label >= 0) & (flattened_label < n)
    return np.bincount(n * flattened_label[k].astype(np.int32) + flattened_pred[k], minlength=n**2).reshape(n, n)


def calculate_batch_hist(labels: list, preds: list, batch_size: int, num_classes: int):
    """
    Args:
        preds(list):
        labels(list):
    Returns:
        output_list(list): in length of batch_size, instance in shape of (H, W, C)

    """
    batch_hist = np.zeros((num_classes, num_classes))
    for idx in range(batch_size):
        pred = preds[idx].flatten()
        gt = labels[idx].flatten()
        batch_hist += calculate_hist(gt, pred, num_classes)
    return batch_hist


def image_preprocess(image, crop_size, args):
    h, w, _ = image.shape
    if h > w:
        new_h = crop_size
        new_w = int(1.0 * crop_size * w / h)
    else:
        new_w = crop_size
        new_h = int(1.0 * crop_size * h / w)
    image = cv2.resize(image, (new_w, new_h))
    resize_h, resize_w, _ = image.shape

    # mean, std
    image = (image - args.image_mean) / args.image_std

    # pad to crop_size
    pad_h = crop_size - image.shape[0]
    pad_w = crop_size - image.shape[1]
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    # hwc to chw
    image = image.transpose((2, 0, 1))
    return image, resize_h, resize_w


def batch_image_preprocess(img_lst, args, crop_size=513):
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    origin_hw = []
    for inner_batch_idx in range(args.batch_size):
        img_ = img_lst[inner_batch_idx]
        origin_hw.append([img_.shape[0], img_.shape[1]])
        img_, resize_h, resize_w = image_preprocess(image=img_, crop_size=crop_size, args=args)
        batch_img[inner_batch_idx] = img_
        resize_hw.append([resize_h, resize_w])
    return batch_img, resize_hw, origin_hw


def resize_to_origin(input, origin_hw, resize_hw, batch_size):
    """
    resize outputs to original size to match labels
    Args:
        input (np.ndarray): a batch of images in shape of (N, C, H, W)
        origin_size(list): image original size to be resized
        batch_size: batch size
    Returns:
        output_list(list): in length of batch_size, instance in shape of (H, W, C)

    """
    output_list = []
    for idx in range(batch_size):
        # print(input[idx][:,:,:].shape)
        res = input[idx][:, : resize_hw[idx][0], : resize_hw[idx][1]].transpose((1, 2, 0))
        # print(res.shape)
        # print("dtype of img to be resized", res.dtype)
        h, w = origin_hw[idx][0], origin_hw[idx][1]
        res = cv2.resize(res, (w, h))
        # print(res.shape)
        output_list.append(res)
    return output_list


def eval_batch(args, eval_net, batch_img, resize_hw, origin_hw, flip=True):
    """batch inference through net"""

    # batch infer through net
    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy().astype(np.float32)

    # if flip, add flipped infer results
    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1].astype(np.float32)

    # resize to origin
    result_lst = resize_to_origin(input=net_out, origin_hw=origin_hw, resize_hw=resize_hw, batch_size=args.batch_size)

    return result_lst


def eval_batch_scales(args, eval_net, img_lst, scales, base_crop_size=513, flip=True):
    """batch inference with muti-scales"""

    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]

    batch_img, resize_hw, origin_hw = batch_image_preprocess(img_lst, args, crop_size=sizes_[0])
    probs_lst = eval_batch(args, eval_net, batch_img, resize_hw, origin_hw, flip=flip)

    for crop_size_ in sizes_[1:]:
        batch_img, resize_hw, origin_hw = batch_image_preprocess(img_lst, args, crop_size=crop_size_)
        probs_lst_tmp = eval_batch(args, eval_net, batch_img, resize_hw, origin_hw, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))

    return result_msk


def apply_eval(eval_param_dict):
    """compute mean IoU"""

    print("evalating...")

    eval_net = eval_param_dict["net"]
    eval_net.set_train(False)
    img_lst = eval_param_dict["dataset"]
    args_ = eval_param_dict["args"]

    args = args_.copy()
    args.batch_size = check_batch_size(len(img_lst), ori_batch_size=args_.batch_size, refine=True)

    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    inner_batch_idx = 0

    for i, line in enumerate(img_lst):
        img_path, msk_path = line.strip().split(" ")
        img_path, msk_path = os.path.join(args.data_root, img_path), os.path.join(args.data_root, msk_path)
        img_, msk_ = cv2.imread(img_path), cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        inner_batch_idx += 1
        if inner_batch_idx == args.batch_size:
            batch_res = eval_batch_scales(
                args, eval_net, batch_img_lst, scales=args.scales, base_crop_size=args.crop_size, flip=args.flip
            )

            hist += calculate_batch_hist(
                labels=batch_msk_lst, preds=batch_res, batch_size=args.batch_size, num_classes=args.num_classes
            )
            inner_batch_idx = 0
            batch_img_lst = []
            batch_msk_lst = []
            if args.eval_processing_log:
                print("processed {} images".format(i + 1))

    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    miou = np.nanmean(iou)
    print("per-class IoU", iou)
    print("mean IoU", np.nanmean(iou))

    return miou
