''' 
Test train and validate pipelines.
For training, both graph mode and pynative mode with ms_function will be tested.
'''
import sys
sys.path.append('.')

import subprocess
from subprocess import Popen, PIPE
import os
import pytest
from mindcv.utils.download import DownLoad

check_acc = True

@pytest.mark.parametrize('mode', ['GRAPH', 'PYNATIVE_FUNC'])
@pytest.mark.parametrize('val_while_train', [True, False])
def test_train(mode, val_while_train,  model='resnet18', opt='adamw', scheduler='polynomial'):
    ''' train on a imagenet subset dataset '''
    # prepare data
    data_dir = 'data/Canidae'
    num_classes = 2
    dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    if not os.path.exists(data_dir):
        DownLoad().download_and_extract_archive(dataset_url, './')

    # ---------------- test running train.py using the toy data ---------
    dataset = 'imagenet'
    ckpt_dir = './tests/ckpt_tmp'
    num_samples = 160
    num_epochs = 5
    batch_size = 20
    if os.path.exists(ckpt_dir):
        os.system(f'rm {ckpt_dir} -rf')
    if os.path.exists(data_dir):
        download_str = f'--data_dir {data_dir}'
    else:
        download_str = '--download'
    train_file = 'train.py' if mode=='GRAPH' else 'train_with_func.py'

    cmd = f'python {train_file} --dataset={dataset} --num_classes={num_classes} --model={model} --epoch_size={num_epochs}  --ckpt_save_interval=2 --lr=0.0001 --num_samples={num_samples} --loss=CE --weight_decay=1e-6 --ckpt_save_dir={ckpt_dir} {download_str} --train_split=train --batch_size={batch_size} --pretrained --num_parallel_workers=2 --val_while_train={val_while_train} --val_split=val --val_interval=1'
    
    print(f'Running command: \n{cmd}')
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret==0, 'Training fails'
    
    # --------- Test running validate.py using the trained model ------------- #
    #begin_ckpt = os.path.join(ckpt_dir, f'{model}-1_1.ckpt')
    end_ckpt = os.path.join(ckpt_dir, f'{model}-{num_epochs}_{num_samples//batch_size}.ckpt')
    cmd = f"python validate.py --model={model} --dataset={dataset} --val_split=val --data_dir={data_dir} --num_classes={num_classes} --ckpt_path={end_ckpt} --batch_size=40 --num_parallel_workers=2"
    #ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    print(f'Running command: \n{cmd}')
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    #assert ret==0, 'Validation fails'
    print(out)

    if check_acc: 
        res = out.decode()
        idx = res.find('Accuracy')
        acc = res[idx:].split(',')[0].split(':')[1]
        print('Val acc: ', acc)
        assert float(acc) > 0.5, 'Acc is too low'
