"""Use the PYNATIVE mode to train the network"""
import os
import logging
from time import time
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, Tensor, ops, SummaryRecord
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import CheckpointManager, Allreduce
from config import parse_args


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
h1 = logging.StreamHandler()
formatter1 = logging.Formatter('%(message)s',)
logger.addHandler(h1)
h1.setFormatter(formatter1)


def train(args):
    """Train network."""
    SEED = 1
    ms.set_seed(SEED)
    np.random.seed(SEED)

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
    
    if args.num_classes is None:
        num_classes = dataset_train.num_classes()
    else:
        num_classes = args.num_classes

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
        cutmix=args.cutmix,
        cutmix_prob=args.cutmix_prob,
        num_classes=num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

    if args.val_while_train:
        dataset_eval = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split=args.val_split,
            num_shards=device_num,
            shard_id=rank_id,
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
        # validation dataset count
        eval_count = dataset_eval.get_dataset_size()
        if args.distribute:
            all_reduce = Allreduce()
            eval_count = all_reduce(Tensor(eval_count, ms.int32))

    num_batches = loader_train.get_dataset_size()
    # Train dataset count
    train_count = dataset_train.get_dataset_size()
    if args.distribute:
        all_reduce = Allreduce()
        train_count = all_reduce(Tensor(train_count, ms.int32))

    # create model
    network = create_model(model_name=args.model,
                           num_classes=num_classes,
                           in_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)

    num_params = sum([param.size for param in network.get_parameters()])

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
                                    decay_rate=args.decay_rate,
                                    milestones=args.multi_step_decay_milestones,
                                    num_epochs=args.epoch_size)

    # resume training if ckpt_path is given
    if args.ckpt_path != '' and args.resume_opt:
        opt_ckpt_path = os.path.join(args.ckpt_save_dir, f'optim_{args.model}.ckpt')
    else:
        opt_ckpt_path = ''

    # create optimizer
    optimizer = create_optimizer(network.trainable_params(),
                                 opt=args.opt,
                                 lr=lr_scheduler,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum,
                                 nesterov=args.use_nesterov,
                                 filter_bias_and_bn=args.filter_bias_and_bn,
                                 loss_scale=args.loss_scale,
                                 checkpoint_path=opt_ckpt_path)

    # resume
    begin_step = 0
    begin_epoch = 0
    if args.ckpt_path != '':
        begin_step = optimizer.global_step.asnumpy()[0]
        begin_epoch = args.ckpt_path.split('/')[-1].split('_')[0].split('-')[-1]
        begin_epoch = int(begin_epoch)

    # log
    if rank_id in [None, 0]:

        logger.info(f"-" * 40)
        logger.info(f"Num devices: {device_num if device_num is not None else 1} \n"
                    f"Distributed mode: {args.distribute} \n"
                    f"Num training samples: {train_count}")
        if args.val_while_train:
            logger.info(f"Num validation samples: {eval_count}")
        logger.info(f"Num classes: {num_classes} \n"
                    f"Num batches: {num_batches} \n"
                    f"Batch size: {args.batch_size} \n"
                    f"Auto augment: {args.auto_augment} \n"
                    f"Model: {args.model} \n"
                    f"Model param: {num_params} \n"
                    f"Num epochs: {args.epoch_size} \n"
                    f"Optimizer: {args.opt} \n"
                    f"LR: {args.lr} \n"
                    f"LR Scheduler: {args.scheduler}")
        logger.info(f"-" * 40)


        assert args.loss_scale == 1.0, 'loss_scale > 1.0 is not supported in train_with_func currently.'


        if args.ckpt_path != '':
            logger.info(f"Resume training from {args.ckpt_path}, last step: {begin_step}, last epoch: {begin_epoch}")
        else:
            logger.info('Start training')

        if not os.path.exists(args.ckpt_save_dir):
            os.makedirs(args.ckpt_save_dir)

        log_path = os.path.join(args.ckpt_save_dir, 'result.log')
        if not (os.path.exists(log_path) and args.ckpt_path != ''):  # if not resume training
            with open(log_path, 'w') as fp:
                fp.write('Epoch\tTrainLoss\tValAcc\tTime\n')

    best_acc = 0
    summary_dir = f"./{args.ckpt_save_dir}/summary_01"

    # Training
    # TODO: args.loss_scale is not making effect.
    need_flush_from_cache = True
    assert (args.ckpt_save_policy != 'top_k' or args.val_while_train == True), \
        "ckpt_save_policy is top_k, val_while_train must be True."
    manager = CheckpointManager(ckpt_save_policy=args.ckpt_save_policy)
    with SummaryRecord(summary_dir) as summary_record:
        for t in range(begin_epoch, args.epoch_size):
            epoch_start = time()

            train_loss = train_epoch(network,
                                     loader_train,
                                     loss,
                                     optimizer,
                                     epoch=t,
                                     n_epochs=args.epoch_size,
                                     summary_record=summary_record,
                                     rank_id=rank_id,
                                     log_interval=args.log_interval)

            # val while train
            test_acc = Tensor(-1.0)
            if args.val_while_train:
                if ((t + 1) % args.val_interval == 0) or (t + 1 == args.epoch_size):
                    if rank_id in [None, 0]:
                        logger.info('Validating...')
                    val_start = time()
                    test_acc = test_epoch(network, loader_eval, rank_id=rank_id)
                    test_acc = 100 * test_acc
                    if rank_id in [0, None]:
                        val_time = time() - val_start
                        logger.info(f"Val time: {val_time:.2f} \t Val acc: {test_acc:0.3f}")
                        if test_acc > best_acc:
                            best_acc = test_acc
                            save_best_path = os.path.join(args.ckpt_save_dir, f"{args.model}-best.ckpt")
                            ms.save_checkpoint(network, save_best_path, async_save=True)
                            logger.info(f"=> New best val acc: {test_acc:0.3f}")

                        # add to summary
                        current_step = (t + 1) * num_batches + begin_step
                        if not isinstance(test_acc, Tensor):
                            test_acc = Tensor(test_acc)
                        if summary_record is not None:
                            summary_record.add_value('scalar', 'test_dataset_accuracy', test_acc)
                            summary_record.record(int(current_step))

            # Save checkpoint
            if rank_id in [0, None]:
                if ((t + 1) % args.ckpt_save_interval == 0) or (t + 1 == args.epoch_size):
                    if need_flush_from_cache:
                        need_flush_from_cache = flush_from_cache(network)

                    ms.save_checkpoint(optimizer, os.path.join(args.ckpt_save_dir, f'{args.model}_optim.ckpt'),
                                       async_save=True)
                    save_path = os.path.join(args.ckpt_save_dir, f"{args.model}-{t + 1}_{num_batches}.ckpt")
                    ckpoint_filelist = manager.save_ckpoint(network, num_ckpt=args.keep_checkpoint_max,
                                                              metric=test_acc, save_path=save_path)
                    if args.ckpt_save_policy == 'top_k':
                        checkpoints_str = "Top K accuracy checkpoints: \n"
                        for ch in ckpoint_filelist:
                            checkpoints_str += '{}\n'.format(ch)
                        logger.info(checkpoints_str)
                    else:
                        logger.info(f"Saving model to {save_path}")

                epoch_time = time() - epoch_start
                logger.info(f'Epoch {t + 1} time:{epoch_time:.3f}s')
                logger.info("-" * 80)
                with open(log_path, 'a') as fp:
                    fp.write(f'{t+1}\t{train_loss.asnumpy():.7f}\t{test_acc.asnumpy():.3f}\t{epoch_time:.2f}\n')

    logger.info("Done!")

def train_epoch(network, dataset, loss_fn, optimizer, epoch, n_epochs, summary_record=None, rank_id=None, log_interval=100):
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
    
    start = time()

    num_batches = dataset.get_dataset_size()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        if args.distribute:
            loss, logits = train_step_parallel(data, label)
        else:
            loss, logits = train_step(data, label)

        if len(label.shape) == 1: 
            correct += (logits.argmax(1) == label).asnumpy().sum()
        else: #one-hot or soft label
            correct += (logits.argmax(1) == label.argmax(1)).asnumpy().sum()
        total += len(data)

        if (batch + 1) % log_interval == 0 or (batch + 1) >= num_batches or batch==0:
            step = epoch * n_batches + batch
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(step)).asnumpy()
            else:
                cur_lr = optimizer.learning_rate.asnumpy()
            logger.info(f"Epoch:[{epoch+1:{epoch_width}d}/{n_epochs:{epoch_width}d}], "
                        f"batch:[{batch+1:{batch_width}d}/{n_batches:{batch_width}d}], "
                        f"loss:{loss.asnumpy():8.6f}, lr: {cur_lr:.7f},  time:{time() - start:.6f}s")
            start = time()
            if rank_id in [0, None]:
                if not isinstance(loss, Tensor):
                    loss = Tensor(loss)
                if summary_record is not None:
                    summary_record.add_value('scalar', 'loss', loss)
                    summary_record.record(step)
                
    if args.distribute:
        all_reduce = Allreduce()
        correct = all_reduce(Tensor(correct, ms.float32))
        total = all_reduce(Tensor(total, ms.float32))
        correct /= total
        correct = correct.asnumpy()
    else:
        correct /= total

    if rank_id in [0, None]:
        logger.info(f"Training accuracy: {(100 * correct):0.3f}")
        if not isinstance(correct, Tensor):
            correct = Tensor(correct)
        if summary_record is not None:
            summary_record.add_value('scalar', 'train_dataset_accuracy', correct)
            summary_record.record(step)
    
    return loss

def test_epoch(network, dataset, rank_id=None):
    """Test network accuracy and loss."""

    #@ms.ms_function
    def test_forward_fn(data):
        logits = network(data)
        return logits

    network.set_train(False) # TODO: check freeze 

    correct, total = 0, 0
    for data, label in tqdm(dataset.create_tuple_iterator()):
        pred = test_forward_fn(data)
        total += len(data)
        if len(label.shape) == 1: 
            correct += (pred.argmax(1) == label).asnumpy().sum()
        else: #one-hot or soft label
            correct += (pred.argmax(1) == label.argmax(1)).asnumpy().sum()

    if rank_id is not None:
        all_reduce = Allreduce()
        correct = all_reduce(Tensor(correct, ms.float32))
        total = all_reduce(Tensor(total, ms.float32))
        correct /= total
        correct = correct.asnumpy()
    else:
        correct /= total

    return correct


def flush_from_cache(network):
    """Flush cache data to host if tensor is cache enable."""
    has_cache_params = False
    params = network.get_parameters()
    for param in params:
        if param.cache_enable:
            has_cache_params = True
            Tensor(param).flush_from_cache()
    if not has_cache_params:
        need_flush_from_cache = False
    else:
        need_flush_from_cache = True
    return need_flush_from_cache

if __name__ == '__main__':
    args = parse_args()
    train(args)
