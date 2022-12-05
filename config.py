import os
import yaml
import logging
import argparse

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
                               help='YAML config file specifying default arguments (default='')')

    # The main parser. It inherits the --config argument for better help information.
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', parents=[parser_config])

    # System parameters
    group = parser.add_argument_group('System parameters')
    group.add_argument('--mode', type=int, default=0,
                       help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    group.add_argument('--distribute', type=str2bool, nargs='?', const=True, default=False,
                       help='Run distribute (default=False)')
    group.add_argument('--val_while_train', type=str2bool, nargs='?', const=True, default=False,
                       help='Verify accuracy while training (default=False)')
    group.add_argument('--val_interval', type=int, default=1,
            help='Interval for validation while training (in epoch), Default: 1')
    group.add_argument('--log_interval', type=int, default=100,
            help='Interval for print training log (in step), if None, log every epoch. Default: 100')
    group.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=False, help='Load pretrained model (default=False)')
    group.add_argument('--ckpt_path', type=str, default='',
                       help='Initialize model from this checkpoint. If resume training, specify the checkpoint path. (default='')')
    group.add_argument('--resume_opt', type=str2bool, nargs='?', const=True, default=False, help='Resume optimizer state including LR (default=False)')


    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--dataset', type=str, default='imagenet',
                       help='Type of dataset (default="imagenet")')
    group.add_argument('--data_dir', type=str, default='./', help='Path to dataset')
    group.add_argument('--train_split', type=str, default='train', help='dataset train split name')
    group.add_argument('--val_split', type=str, default='val', help='dataset validation split name')
    group.add_argument('--dataset_download', type=str2bool, nargs='?', const=True, default=False,
                       help='Download dataset (default=False)')
    group.add_argument('--num_parallel_workers', type=int, default=8,
                       help='Number of parallel workers (default=8)')
    group.add_argument('--shuffle', type=str2bool, nargs='?', const=True, default=True,
                       help='Whether or not to perform shuffle on the dataset (default="True")')
    group.add_argument('--num_samples', type=int, default=None,
                       help='Number of elements to sample (default=None, which means sample all elements)')
    group.add_argument('--batch_size', type=int, default=128,
                       help='Number of batch size (default=128)')
    group.add_argument('--drop_remainder', type=str2bool, nargs='?', const=True, default=True,
                       help='Determines whether or not to drop the last block whose data '
                            'row number is less than batch size (default=True)')

    # Augmentation parameters
    group = parser.add_argument_group('Augmentation parameters')
    group.add_argument('--image_resize', type=int, default=224,
                       help='Crop the size of the image (default=224)')
    group.add_argument('--scale', type=tuple, default=(0.08, 1.0),
                       help='Random resize scale (default=(0.08, 1.0))')
    group.add_argument('--ratio', type=tuple, default=(0.75, 1.333),
                       help='Random resize aspect ratio (default=(0.75, 1.333))')
    group.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability (default=0.5)')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability (default=0.)')
    group.add_argument('--color_jitter', type=float, default=0.4,
                       help='Color jitter factor (default=None)')
    group.add_argument('--interpolation', type=str, default='bilinear',
                       help='Image interpolation mode for resize operator(default="bilinear")')
    group.add_argument('--auto_augment', type=str, default=None,
            help='Auto augment policy config. If "randaug",apply RandAugment, "autoaug" for original AutoAugment, "autoaugr" for AutoAugment with increasing posterize, or None. If apply, recommend for imagenet: randaug-m7-mstd0.5 (defalt: None).'
                            'Example: "randaug-m10-n2-w0-mstd0.5-mmax10-inc0", "autoaug-mstd0.5" or autoaugr-mstd0.5.')
    group.add_argument('--re_prob', type=float, default=0.,
                       help='Probability of performing erasing (default=0.)')
    group.add_argument('--re_scale', type=tuple, default=(0.02, 0.33),
                       help='Range of area scale of the erased area (default=(0.02, 0.33))')
    group.add_argument('--re_ratio', type=tuple, default=(0.3, 3.3),
                       help='Range of aspect ratio of the erased area (default=(0.3, 3.3))')
    group.add_argument('--re_value', default=0,
                       help='Pixel value used to pad the erased area (default=0)')
    group.add_argument('--re_max_attempts', type=int, default=10,
                       help='The maximum number of attempts to propose a valid erased area, '
                            'beyond which the original image will be returned (default=10)')
    group.add_argument('--mean', type=list, default=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                       help='List or tuple of mean values for each channel, '
                            'with respect to channel order (default=[0.485 * 255, 0.456 * 255, 0.406 * 255])')
    group.add_argument('--std', type=list, default=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                       help='List or tuple of mean values for each channel, '
                            'with respect to channel order (default=[0.229 * 255, 0.224 * 255, 0.225 * 255])')
    group.add_argument('--crop_pct', type=float, default=0.875,
                       help='Input image center crop percent (default=0.875)')
    group.add_argument('--cutmix', type=float, default=0.,
                       help='Hyperparameter of beta distribution of cutmix. (default=0.)')
    group.add_argument('--cutmix_prob', type=float, default=1.0,
                       help='probability of applying cutmix and/or mixup (default=0.)')
    group.add_argument('--mixup', type=float, default=0.,
                       help='Hyperparameter of beta distribution of mixup. Recommended value is 0.2 for ImageNet. (default=0.)')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', type=str, default='mobilenet_v2_035_224',
                       help='Name of model')
    group.add_argument('--num_classes', type=int, default=None,
                       help='Number of label classes. If None, read from standard datasets. (default=None)')
    group.add_argument('--drop_rate', type=float, default=None,
                       help='Drop rate (default=None)')
    group.add_argument('--drop_path_rate', type=float, default=None,
                       help='Drop path rate (default=None)')
    group.add_argument('--amp_level', type=str, default='O0', help='Amp level - Auto Mixed Precision level for saving memory and acceleration. choice: O0 - all FP32, O1 - only cast ops in white-list to FP16, O2 - cast all ops except for blacklist to FP16, O3 - cast all ops to FP16.   (default="O0").')
    group.add_argument('--keep_checkpoint_max', type=int, default=10,
                       help='Max number of checkpoint files (default=10)')
    group.add_argument('--ckpt_save_dir', type=str, default="./ckpt",
                       help='Path of checkpoint (default="./ckpt")')
    group.add_argument('--ckpt_save_interval', type=int, default=1,
                       help='checkpoint saving interval, unit: epoch, (default=1)')
    group.add_argument('--epoch_size', type=int, default=90,
                       help='Train epoch size (default=90)')
    group.add_argument('--dataset_sink_mode', type=str2bool, nargs='?', const=True, default=True,
                       help='The dataset sink mode (default=True).')
    group.add_argument('--in_channels', type=int, default=3,
                       help='Input channels (default=3)')
    group.add_argument('--ckpt_save_policy', type=str, default='latest_k',
                       help='Checkpoint saving strategy. The optional values is None, "top_k" or "latest_k".')

    # Optimize parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', type=str, default='adam',
                       choices=['sgd', 'momentum', 'adam', 'adamw', 'rmsprop', 'adagrad', 'lamb', "nadam"],
                       help='Type of optimizer (default="adam")')
    group.add_argument('--momentum', type=float, default=0.9,
                       help='Hyperparameter of type float, means momentum for the moving average. '
                            'It must be at least 0.0 (default=0.9)')
    group.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay (default=1e-6)')
    #group.add_argument('--loss_scaler', type=str, default='static', help='Loss scaler, static or dynamic (default=static)')
    group.add_argument('--loss_scale', type=float, default=1.0,
                       help='Loss scale (default=1.0)')
    group.add_argument('--use_nesterov', type=str2bool, nargs='?', const=True, default=False,
                       help='Enables the Nesterov momentum (default=False)')
    group.add_argument('--filter_bias_and_bn', type=str2bool, nargs='?', const=True, default=True,
                       help='Filter Bias and BatchNorm (default=True)')
    group.add_argument('--eps', type=float, default=1e-10,
                       help='Term Added to the Denominator to Improve Numerical Stability (default=1e-10)')

    # Scheduler parameters
    group = parser.add_argument_group('Scheduler parameters')
    group.add_argument('--scheduler', type=str, default='cosine_decay',
                       choices=['constant', 'cosine_decay', 'exponential_decay', 'step_decay', 'multi_step_decay'],
                       help='Type of scheduler (default="warmup_consine_decay")')
    group.add_argument('--lr', type=float, default=0.001,
                       help='learning rate (default=0.001)')
    group.add_argument('--min_lr', type=float, default=1e-6,
                       help='The minimum value of learning rate if scheduler supports (default=None)')
    group.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs (default=None)')
    group.add_argument('--warmup_factor', type=float, default=0.0,
                       help='Warmup factor of learning rate (default=0.0)')
    group.add_argument('--decay_epochs', type=int, default=100,
                       help='Decay epochs (default=None)')
    group.add_argument('--decay_rate', type=float, default=0.9,
                       help='LR decay rate if scheduler supports')
    group.add_argument('--multi_step_decay_milestones', type=list, default=[30, 60, 90],
                       help='list of epoch milestones for lr decay, which is ONLY effective for the multi_step_decay scheduler. LR will be decay by decay_rate at the milestone epoch.')
    group.add_argument('--lr_epoch_stair', type=str2bool, nargs='?', const=True, default=False,
                       help='If True, LR will be updated in the begin of each new epoch and the LR will be consisent for each batch in one epoch. Otherwise, learning rate will be updated dynamically in each step. (default=False)')

    # Loss parameters
    group = parser.add_argument_group('Loss parameters')
    # fixme: add options, bce_loss, ce_loss
    group.add_argument('--loss', type=str, default='CE', choices=['BCE', 'CE'],
                       help='Type of loss, BCE (BinaryCrossEntropy) or CE (CrossEntropy)  (default="CE")')
    group.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Use label smoothing (default=0.0)')
    group.add_argument('--aux_factor', type=float, default=0.0,
                       help='Aux loss factor (default=0.0)')
    group.add_argument('--reduction', type=str, default='mean',
                       help='Type of reduction to be applied to loss (default="mean")')

    return parser_config, parser


def parse_args():
    parser_config, parser = create_parser()
    # Do we have a config file to parse?
    args_config, remaining = parser_config.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


def save_args(args: argparse.Namespace, filepath: str, rank: int = 0) -> None:
    """If in master process, save ``args`` to a YAML file. Otherwise, do nothing.
    Args:
        args (Namespace): The parsed arguments to be saved.
        filepath (str): A filepath ends with ``.yaml``.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert isinstance(args, argparse.Namespace)
    assert filepath.endswith(".yaml")
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.safe_dump(args.__dict__, f)
    logger.info(f"Args is saved to {filepath}.")
