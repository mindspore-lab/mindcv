# system config
mode: 0
distribute: True
num_parallel_workers: 8
val_while_train: True

# dataset config
dataset: 'imagenet'
data_dir: ''
shuffle: True
dataset_download: False
batch_size: 128
drop_remainder: True

# Augmentation config
image_resize: 224
scale: [0.08, 1.0]
ratio: [0.75, 1.333]
hflip: 0.5
vflip: 0.0
interpolation: 'bilinear'
crop_pct: 0.9
color_jitter: [0.4, 0.4, 0.4]
re_prob: 0.25
mixup: 0.8
cutmix_prob: 1.0
switch_prob: 0.5
mixup_mode: batch
cutmix: 1.0
auto_augment: 'randaug-m9-mstd0.5-inc1'

# model config
model: 'poolformer_s12'
drop_rate: 0.0
drop_path_rate: 0.1
num_classes: 1000
pretrained: False
ckpt_path: ''
keep_checkpoint_max: 10
ckpt_save_dir: './'
epoch_size: 400
dataset_sink_mode: True
amp_level: 'O0'

# loss config
loss: 'CE'
label_smoothing: 0.1

# lr scheduler config
scheduler: 'warmup_cosine_decay' 
min_lr: 1.0e-05
lr: 0.001
warmup_epochs: 5
decay_epochs: 395
decay_rate: 0.1

# optimizer config
opt: 'AdamW'
filter_bias_and_bn: True
momentum: 0.9
weight_decay: 0.05
loss_scale: 1024
use_nesterov: False
