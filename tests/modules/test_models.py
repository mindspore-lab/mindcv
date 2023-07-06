import sys

sys.path.append(".")

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindcv import list_models, list_modules
from mindcv.models import (
    create_model,
    get_pretrained_cfg_value,
    is_model_in_modules,
    is_model_pretrained,
    model_entrypoint,
)

# TODO: the global avg pooling op used in EfficientNet is not supported for CPU.
# TODO: memory resource is limited on free github action runner, ask the PM for self-hosted runners!
model_name_list = [
    "BiT_resnet50",
    "repmlp_t224",
    "convit_tiny",
    "convnext_tiny",
    "crossvit_9",
    "densenet121",
    "dpn92",
    "edgenext_small",
    "ghostnet_100",
    "googlenet",
    "hrnet_w32",
    "inception_v3",
    "inception_v4",
    "mixnet_s",
    "mnasnet_050",
    "mobilenet_v1_025",
    "mobilenet_v2_035_128",
    "mobilenet_v3_small_075",
    "nasnet_a_4x1056",
    "pnasnet",
    "poolformer_s12",
    "pvt_tiny",
    "pvt_v2_b0",
    "regnet_x_200mf",
    "repvgg_a0",
    "res2net50",
    "resnet18",
    "resnext50_32x4d",
    "rexnet_09",
    "seresnet18",
    "shufflenet_v1_g3_05",
    "shufflenet_v2_x0_5",
    "skresnet18",
    "squeezenet1_0",
    "swin_tiny",
    "visformer_tiny",
    "vit_b_32_224",
    "xception",
]

check_loss_decrease = False


# @pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("name", model_name_list)
def test_model_forward(name):
    # ms.set_context(mode=ms.PYNATIVE_MODE)
    bs = 2
    c = 10
    model = create_model(model_name=name, num_classes=c)
    input_size = get_pretrained_cfg_value(model_name=name, cfg_key="input_size")
    if input_size:
        input_size = (bs,) + tuple(input_size)
    else:
        input_size = (bs, 3, 224, 224)
    dummy_input = Tensor(np.random.rand(*input_size), dtype=ms.float32)
    y = model(dummy_input)
    assert y.shape == (bs, 10), "output shape not match"


"""
@pytest.mark.parametrize('name', model_name_list)
def test_model_backward(name):
    # TODO: check number of gradient == number of parameters
    bs = 8
    c = 2
    input_data = Tensor(np.random.rand(bs, 3, 224, 224), dtype=ms.float32)
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
"""


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
    assert ouptput_false_list == [], "{}\n, Above mentioned models do not exist within a subset of modules.".format(
        ouptput_false_list
    )


def test_is_model_pretrained():
    model_name_list = list_models()
    ouptput_false_list = []
    num_pretrained = 0
    for model_name in model_name_list:
        if not is_model_pretrained(model_name):
            ouptput_false_list.append(model_name)
        else:
            num_pretrained += 1
    # assert ouptput_false_list == [], \
    #    '{}\n, Above mentioned models do not have pretrained models.'.format(ouptput_false_list)

    assert num_pretrained > 0, "No pretrained models"


if __name__ == "__main__":
    test_model_forward("pnasnet")
    """
    for model in model_name_list:
        if '384' in model:
            print(model)
            test_model_forward(model)
    """
