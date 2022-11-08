import sys
sys.path.append('.')

import pytest

import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore
from mindcv import list_models, list_modules
from mindcv.models import create_model, model_entrypoint, is_model_in_modules, is_model_pretrained
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindspore.nn import TrainOneStepCell, WithLossCell
import math

#TODO: the global avg pooling op used in EfficientNet is not supported for CPU. 
model_name_list = list_models(exclude_filters='efficientnet*')

check_loss_decrease = False

#@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize('name', model_name_list)
def test_model_forward(name):
    #ms.set_context(mode=ms.GRAPH_MODE)
    bs = 8
    c = 10
    model= create_model(model_name=name, num_classes=c)
    if hasattr(model, 'image_size'):
        image_size = model.image_size
    else:
        image_size = 224
    dummy_input = Tensor(np.random.rand(bs, 3, image_size, image_size), dtype=mindspore.float32)
    y = model(dummy_input)
    assert y.shape==(bs, 10), 'output shape not match'

'''
@pytest.mark.parametrize('name', model_name_list)
def test_model_backward(name):
    # TODO: check number of gradient == number of parameters
    bs = 8
    c = 2
    input_data = Tensor(np.random.rand(bs, 3, 224, 224), dtype=mindspore.float32)
    label = Tensor(np.random.randint(0, high=c, size=(bs)), dtype=ms.int32)

    model= create_model(model_name=name, num_classes=c)

    net_loss = create_loss(name='CE')
    net_opt = create_optimizer(model.trainable_params(), 'adam', lr=0.0001)
    net_with_loss = WithLossCell(model, net_loss)

    train_network = TrainOneStepCell(net_with_loss, net_opt)

    begin_loss = train_network(input_data, label)
    for _ in range(2):
        cur_loss = train_network(input_data, label)
    print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))
    
    assert not math.isnan(cur_loss), 'loss NaN when training {name}' 
    if check_loss_decrease:
        assert cur_loss < begin_loss, 'Loss does NOT decrease'
'''

def test_list_models():
    model_name_list = list_models()
    for model_name in model_name_list:
        print(model_name)


def test_model_entrypoint():
    model_name_list = list_models()
    for model_name in model_name_list:
        print(model_entrypoint(model_name))

def test_list_modules():
    module_name_list = list_modules()
    for module_name in module_name_list:
        print(module_name)

def test_is_model_in_modules():
    model_name_list = list_models()
    module_names = list_modules()
    ouptput_false_list = []
    for model_name in model_name_list:
        if not is_model_in_modules(model_name, module_names):
            ouptput_false_list.append(model_name)
    assert ouptput_false_list == [], \
        '{}\n, Above mentioned models do not exist within a subset of modules.'.format(ouptput_false_list)
    

def test_is_model_pretrained():
    model_name_list = list_models()
    ouptput_false_list = []
    num_pretrained = 0
    for model_name in model_name_list:
        if not is_model_pretrained(model_name):
            ouptput_false_list.append(model_name)
        else:
            num_pretrained += 1
    #assert ouptput_false_list == [], \
    #    '{}\n, Above mentioned models do not have pretrained models.'.format(ouptput_false_list)

    assert num_pretrained > 0, 'No pretrained models'

if __name__== '__main__':
    for model in model_name_list: 
        if '384' in model:
            print(model)
            test_model_forward(model)
