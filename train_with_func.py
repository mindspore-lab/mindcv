"""Use the PYNATIVE mode to train the network"""
import os
from time import time

import mindspore as ms
from mindspore import nn, Tensor, ops, SummaryRecord
from mindspore.ops import ReduceOp
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from config import parse_args


def train_epoch(network, dataset, loss_fn, optimizer, epoch, n_epochs, summary_record, rank_id=None, log_interval=100):
    """Training an epoch network"""
    # Define forward function
    def forward_fn(data, label):
        logits = network(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    if args.distribute:
        mean = _get_gradients_mean()
        degree = _get_device_num()
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    # Define function of one-step training
    @ms.ms_function
    def train_step_parallel(data, label):
        (loss, logits), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    @ms.ms_function
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    network.set_train()
    n_batches = dataset.get_dataset_size()
    n_steps = n_batches * n_epochs
    epoch_width, batch_width, step_width = len(str(n_epochs)), len(str(n_batches)), len(str(n_steps))
    total, correct = 0, 0
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        step_time = time()
        if args.distribute:
            loss, logits = train_step_parallel(data, label)
        else:
            loss, logits = train_step(data, label)
        correct += (logits.argmax(1) == label).sum().asnumpy()
        total += len(data)

        if (batch + 1) % log_interval == 0:
            step = epoch * n_batches + batch + 1
            print(f"Epoch:[{epoch+1:{epoch_width}d}/{n_epochs:{epoch_width}d}], "
                  f"batch:[{batch+1:{batch_width}d}/{n_batches:{batch_width}d}], "
                  f"step:[{step:{step_width}d}/{n_steps:{step_width}d}], "
                  f"loss:{loss.asnumpy():8.6f}, time:{time() - step_time:.6f}s")

            if (not args.distribute) or (args.distribute and rank_id == 0):
                if not isinstance(loss, Tensor):
                    loss = Tensor(loss)
                summary_record.add_value('scalar', 'loss', loss)
                summary_record.record(step)

    if args.distribute:
        all_reduce = ops.AllReduce(ReduceOp.SUM)
        correct = all_reduce(Tensor(correct, ms.float32))
        total = all_reduce(Tensor(total, ms.float32))
        correct /= total
        correct = correct.asnumpy()
    else:
        correct /= total

    if (not args.distribute) or (args.distribute and rank_id == 0):
        print(f"Train Dataset: Epoch:[{epoch+1:{epoch_width}d}/{n_epochs:{epoch_width}d}],"
              f"Accuracy: {(100 * correct):>0.1f}%")
        if not isinstance(correct, Tensor):
            correct = Tensor(correct)
        summary_record.add_value('scalar', 'train_dataset_accuracy', correct)
        summary_record.record(step)

def test_loop(network, dataset, loss_fn, rank_id=None):
    """Test network accuracy and loss."""
    num_batches = dataset.get_dataset_size()

    network.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = network(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).sum().asnumpy()

    if rank_id is not None:
        all_reduce = ops.AllReduce(ReduceOp.SUM)
        test_loss = all_reduce(Tensor(test_loss, ms.float32))
        num_batches = all_reduce(Tensor(num_batches, ms.float32))
        correct = all_reduce(Tensor(correct, ms.float32))
        total = all_reduce(Tensor(total, ms.float32))
        test_loss /= num_batches
        correct /= total
        test_loss = test_loss.asnumpy()
        correct = correct.asnumpy()
    else:
        test_loss /= num_batches
        correct /= total

    return correct, test_loss


def train(args):
    """Train network."""
    ms.set_seed(1)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = None
        rank_id = None

    # create dataset
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.train_split,
        shuffle=args.shuffle,
        num_samples=args.num_samples,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download)

    # create transforms
    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=True,
        image_resize=args.image_resize,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        mean=args.mean,
        std=args.std,
        re_prob=args.re_prob,
        re_scale=args.re_scale,
        re_ratio=args.re_ratio,
        re_value=args.re_value,
        re_max_attempts=args.re_max_attempts
    )

    # load dataset
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=False,
        is_training=True,
        mixup=args.mixup,
        num_classes=args.num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

    if args.val_while_train:
        dataset_eval = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split=args.val_split,
            num_parallel_workers=args.num_parallel_workers,
            download=args.dataset_download)

        transform_list_eval = create_transforms(
            dataset_name=args.dataset,
            is_training=False,
            image_resize=args.image_resize,
            crop_pct=args.crop_pct,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std
        )

        loader_eval = create_loader(
            dataset=dataset_eval,
            batch_size=args.batch_size,
            drop_remainder=False,
            is_training=False,
            transform=transform_list_eval,
            num_parallel_workers=args.num_parallel_workers,
        )

    num_batches = loader_train.get_dataset_size()

    # create model
    network = create_model(model_name=args.model,
                           num_classes=args.num_classes,
                           in_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)

    # create loss
    loss = create_loss(name=args.loss,
                       reduction=args.reduction,
                       label_smoothing=args.label_smoothing,
                       aux_factor=args.aux_factor)

    # create learning rate schedule
    lr_scheduler = create_scheduler(num_batches,
                                    scheduler=args.scheduler,
                                    lr=args.lr,
                                    min_lr=args.min_lr,
                                    warmup_epochs=args.warmup_epochs,
                                    decay_epochs=args.decay_epochs,
                                    decay_rate=args.decay_rate)

    # create optimizer
    optimizer = create_optimizer(network.trainable_params(),
                                 opt=args.opt,
                                 lr=lr_scheduler,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum,
                                 nesterov=args.use_nesterov,
                                 filter_bias_and_bn=args.filter_bias_and_bn,
                                 loss_scale=args.loss_scale)

    # training
    # TODO: args.loss_scale is not making effect.
    print('Training...')
    best_acc = 0
    summary_dir = "./summary_dir/summary_01"
    with SummaryRecord(summary_dir) as summary_record:
        for t in range(args.epoch_size):
            epoch_time = time()
            train_epoch(network, loader_train, loss, optimizer, epoch=t, n_epochs=args.epoch_size,
                        summary_record=summary_record, rank_id=rank_id, log_interval=num_batches)
            print(f'Epoch {t + 1} training time: {time() - epoch_time:.3f}s')

            # Save checkpoint
            if ((t + 1) % args.ckpt_save_interval == 0) or (t+1 == args.epoch_size):
                if (not args.distribute) or (args.distribute and rank_id == 0):
                    os.makedirs(args.ckpt_save_dir, exist_ok=True)
                    save_path = os.path.join(args.ckpt_save_dir, f"{args.model}-{t + 1}_{num_batches}.ckpt") # for consistency with train.py
                    ms.save_checkpoint(network, save_path, async_save=True)
                    print(f"Saving model to {save_path}")

            if args.val_while_train:
                if ((t + 1) % args.ckpt_save_interval == 0) or (t+1 == args.epoch_size):
                    test_acc, test_loss = test_loop(network, loader_eval, loss, rank_id=rank_id)
                    if (not args.distribute) or (args.distribute and rank_id == 0):
                        print(f"Test Dataset: Epoch: {t + 1}, Accuracy: {(100 * test_acc):>0.1f}%,"
                              f"Avg loss: {test_loss:>8f}")
                        current_step = (t + 1) * num_batches
                        if not isinstance(test_acc, Tensor):
                            test_acc = Tensor(test_acc)
                        summary_record.add_value('scalar', 'test_dataset_accuracy', test_acc)
                        summary_record.record(current_step)

                        if test_acc > best_acc:
                            best_acc = test_acc
                            save_best_path = os.path.join(args.ckpt_save_dir, f"{args.model}-best.ckpt")
                            ms.save_checkpoint(network, save_best_path, async_save=True)
                            print(f"Saving best accuracy model to {save_best_path}")
                        print("-" * 80)

    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    train(args)
