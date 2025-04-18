import mindspore as ms
import mindspore.nn as nn
from mindspore import Model
from mindspore.dataset import GeneratorDataset

from mindcv.data.token_data import ImageTokenDataset
from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.utils import ValCallback
from mindcv.utils.top_k import Top1CategoricalAccuracyForTokenData, Top5CategoricalAccuracyForTokenData

from config import parse_args  # isort: skip


def validate(args):
    ms.set_context(mode=args.mode)

    # TODO: move it inside
    pad_info = {
        "token": ([args.max_seq_length, 3 * args.patch_size * args.patch_size], 0),
        "pos": ([args.max_seq_length, 2], 0),
        "ind": ([args.max_seq_length], -1),
        "y": ([args.max_num_each_group], -1),  # ignore_index in loss function
    }

    # create dataset
    dataset_eval = ImageTokenDataset(
        args.data_dir,
        split=args.val_split,
        patch_size=args.patch_size,
        max_seq_length=args.max_seq_length,
        enable_cache=True,
        cache_path="val_db_info.json",
        interpolation=args.interpolation,
        image_resize=args.image_resize,
        max_num_each_group=args.max_num_each_group,
    )

    # create transform
    loader_eval = GeneratorDataset(
        dataset_eval,
        column_names=["token", "pos", "ind", "y"],
        num_parallel_workers=args.num_parallel_workers,
        max_rowsize=12,
        shuffle=False,
    )
    loader_eval = loader_eval.padded_batch(args.batch_size, drop_remainder=False, pad_info=pad_info)

    # read num clases
    num_classes = dataset_eval.num_classes() if args.num_classes is None else args.num_classes

    # create model
    network = create_model(
        model_name=args.model,
        num_classes=num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        pretrained=args.pretrained,
        checkpoint_path=args.ckpt_path,
        ema=args.ema,
        max_num_each_group=args.max_num_each_group,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.val_amp_level)

    # create loss
    loss = create_loss(
        name=args.loss,
        reduction=args.reduction,
        label_smoothing=args.label_smoothing,
        aux_factor=args.aux_factor,
    )

    # Define eval metrics.
    if num_classes >= 5:
        eval_metrics = {
            "Top_1_Accuracy": Top1CategoricalAccuracyForTokenData(),
            "Top_5_Accuracy": Top5CategoricalAccuracyForTokenData(),
            "loss": nn.metrics.Loss(),
        }
    else:
        eval_metrics = {
            "Top_1_Accuracy": Top1CategoricalAccuracyForTokenData(),
            "loss": nn.metrics.Loss(),
        }

    # init model
    model = Model(network, loss_fn=loss, metrics=eval_metrics)

    # log
    num_batches = loader_eval.get_dataset_size()
    print(f"Model: {args.model}")
    print(f"Num batches: {num_batches}")
    print("Start validating...")

    # validate
    result = model.eval(loader_eval, dataset_sink_mode=False, callbacks=[ValCallback(args.log_interval)])
    print(result)


if __name__ == "__main__":
    args = parse_args()
    validate(args)
