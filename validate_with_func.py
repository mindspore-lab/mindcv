""" ImageNet Validation Script

Example:
    $ python validate_with_func.py --model=densenet121 --data_dir="/path/to/data" --pretrained
"""

from tqdm import tqdm
import mindspore as ms
from mindspore import ops

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from config import parse_args


def validate(model, dataset, loss_fn):
    """Evaluates model on validation data with top-1 & top-5 metrics."""
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, acc1, acc5 = 0, 0, 0, 0
    for data, label in tqdm(dataset.create_tuple_iterator(), total=num_batches):
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        acc1 += ops.intopk(pred, label, 1).sum().asnumpy()
        acc5 += ops.intopk(pred, label, 5).sum().asnumpy()
    test_loss /= num_batches
    acc1 /= total
    acc5 /= total
    return acc1, acc5, test_loss


def main():
    args = parse_args()
    ms.set_seed(1)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # create dataset
    dataset_eval = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.val_split,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download)

    # create transform
    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=False,
        image_resize=args.image_resize,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std
    )

    # load dataset
    loader_eval = create_loader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        drop_remainder=False,
        is_training=False,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

    num_classes = dataset_eval.num_classes() if args.num_classes==None else args.num_classes

    # create model
    network = create_model(model_name=args.model,
                           num_classes=num_classes,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)
    network.set_train(False)

    # create loss
    loss = create_loss(name=args.loss,
                       reduction=args.reduction,
                       label_smoothing=args.label_smoothing,
                       aux_factor=args.aux_factor)

    # validate
    print("Testing...")
    test_acc1, test_acc5, test_loss = validate(network, loader_eval, loss)
    print(f"Acc@1: {(100 * test_acc1):>0.1f}%, Acc@5: {(100 * test_acc5):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    main()
