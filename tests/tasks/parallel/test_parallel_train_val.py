''' 
Test training in parallel.
For training, both graph mode and pynative mode with ms_function will be tested.
'''
import sys
sys.path.append('.')

import subprocess
from subprocess import Popen, PIPE
import os
import pytest
from mindcv.utils.download import DownLoad
import glob

check_acc = True

@pytest.mark.parametrize('mode', ['GRAPH', 'PYNATIVE_FUNC'])
def test_train(mode,  model='resnet18', opt='adamw', scheduler='polynomial'):
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
    num_epochs = 5
    batch_size = 10
    num_samples = 160
    if os.path.exists(ckpt_dir):
        os.system(f'rm {ckpt_dir} -rf')
    if os.path.exists(data_dir):
        download_str = f'--data_dir {data_dir}'
    else:
        download_str = '--download'
    # pick gpu devices for parallel
    #cmd = 'export CUDA_VISIBLE_DEVICES=0,1,2,3'
    #subprocess.call(cmd, shell=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # train
    train_file = 'train.py' if mode=='GRAPH' else 'train_with_func.py' 
    cmd = f'mpirun --allow-run-as-root -n 2 python {train_file} --dataset={dataset} --num_classes={num_classes} --model={model} --epoch_size={num_epochs}  --ckpt_save_interval=2 --lr=0.0001 --loss=CE --weight_decay=1e-6 --ckpt_save_dir={ckpt_dir} {download_str} --train_split=train --batch_size={batch_size} --pretrained --distribute'
    
    print(f'Running command: \n{cmd}')
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret==0, 'Training fails'
    
    # --------- Test running validate.py using the trained model ------------- #
    #begin_ckpt = os.path.join(ckpt_dir, f'{model}-1_1.ckpt')
    end_ckpt = os.path.join(ckpt_dir, f'{model}-{num_epochs}_*.ckpt')
    end_ckpt = glob.glob(end_ckpt)[-1]
    cmd = f"python validate.py --model={model} --dataset={dataset} --val_split=val --data_dir={data_dir} --num_classes={num_classes} --ckpt_path={end_ckpt} --batch_size=40"
    #ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    print(f'Running command: \n{cmd}')
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    #assert ret==0, 'Validation fails'
    print(out)

    if check_acc: 
        res = out.decode()
        acc = res.split(',')[0].split(':')[1]
        print('Val acc: ', acc)
        assert float(acc) > 0.5, 'Acc is too low'

if __name__=="__main__":
    test_train('GRAPH', 'resnet18', 'adam', 'polynomial_decay')
