import sys
sys.path.append('.')

import pytest

import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore
from mindcv import list_models, list_modules
from mindcv.models import create_model, model_entrypoint, is_model_in_modules, is_model_pretrained

ms.set_context(mode=ms.PYNATIVE_MODE)

model_name_list = list_models()

@pytest.mark.parametrize('name', model_name_list)
def test_models(name):

    bs = 8
    c = 10
    model= create_model(model_name=name, num_classes=c)
    dummy_input = Tensor(np.random.rand(bs, 3, 224, 224), dtype=mindspore.float32)
    y = model(dummy_input)
    assert y.shape==(bs, 10), 'output shape not match'


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
    for model_name in model_name_list:
        if not is_model_pretrained(model_name):
            ouptput_false_list.append(model_name)
    assert ouptput_false_list == [], \
        '{}\n, Above mentioned models do not have pretrained models.'.format(ouptput_false_list)