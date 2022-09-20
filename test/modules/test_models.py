import sys
sys.path.append('.')

import pytest

import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.common.initializer import Normal
import mindspore
from mindcv import list_models
from mindcv.models import create_model

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

   
