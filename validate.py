import mindspore as ms
import mindspore.nn as nn
from mindspore import Model

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from config import parse_args
from mindcv.utils.utils import check_batch_size
from mindcv.utils.callbacks import ValCallback


def validate(args):
    ms.set_context(mode=args.mode)

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
    
    # read num clases
    num_classes = dataset_eval.num_classes() if args.num_classes==None else args.num_classes
    
    # check batch size
    batch_size = check_batch_size(dataset_eval.get_dataset_size(), args.batch_size)
        
    # load dataset
    loader_eval = create_loader(
        dataset=dataset_eval,
        batch_size=batch_size,
        drop_remainder=False,
        is_training=False,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

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

    # Define eval metrics.
    if num_classes >= 5:
        eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                        'Top_5_Accuracy': nn.Top5CategoricalAccuracy(),
                        'loss': nn.metrics.Loss()
                        }
    else:
        eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(), 
                        'loss': nn.metrics.Loss()}

    # init model
    model = Model(network, loss_fn=loss, metrics=eval_metrics)
    
    # log
    num_batches = loader_eval.get_dataset_size()
    print(f"Model: {args.model}")
    print(f"Num batches: {num_batches}")
    print(f"Start validating...")

    # validate
    result = model.eval(loader_eval, 
            dataset_sink_mode=False,
            callbacks=[ValCallback(args.log_interval)])
    print(result)


if __name__ == '__main__':
    args = parse_args()
    validate(args)
