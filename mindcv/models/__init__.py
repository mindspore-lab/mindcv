"""models init"""
from . import layers, convnext, densenet, dpn, efficientnet, ghostnet, googlenet, inception_v3, inception_v4, mnasnet,\
    mobilenet_v1, mobilenet_v2, mobilenet_v3, model_factory, nasnet, pnasnet, registry, repvgg, res2net, resnet,\
    rexnet, shufflenetv1, shufflenetv2, sknet, squeezenet, swin_transformer, vgg, xception, convit

__all__ = []
__all__.extend(layers.__all__)
__all__.extend(convnext.__all__)
__all__.extend(densenet.__all__)
__all__.extend(dpn.__all__)
__all__.extend(efficientnet.__all__)
__all__.extend(ghostnet.__all__)
__all__.extend(googlenet.__all__)
__all__.extend(inception_v3.__all__)
__all__.extend(inception_v4.__all__)
__all__.extend(mnasnet.__all__)
__all__.extend(mobilenet_v1.__all__)
__all__.extend(mobilenet_v2.__all__)
__all__.extend(mobilenet_v3.__all__)
__all__.extend(model_factory.__all__)
__all__.extend(nasnet.__all__)
__all__.extend(pnasnet.__all__)
__all__.extend(registry.__all__)
__all__.extend(repvgg.__all__)
__all__.extend(res2net.__all__)
__all__.extend(resnet.__all__)
__all__.extend(rexnet.__all__)
__all__.extend(shufflenetv1.__all__)
__all__.extend(shufflenetv2.__all__)
__all__.extend(sknet.__all__)
__all__.extend(squeezenet.__all__)
__all__.extend(swin_transformer.__all__)
__all__.extend(vgg.__all__)
__all__.extend(xception.__all__)
__all__.extend(convit.__all__)

#fixme: since googlenet is used as both the file and function name, we need to import * after __all__ 

from .densenet import *
from .googlenet import *
from .inception_v3 import *
from .inception_v4 import *
from .mobilenet_v1 import *
from .mobilenet_v2 import *
from .mobilenet_v3 import *
from .xception import *
from .model_factory import *
from .registry import *
from .resnet import *
from .rexnet import *
from .shufflenetv1 import *
from .shufflenetv2 import *
from .squeezenet import *
from .mnasnet import *
from .sknet import *
from .res2net import *
from .utils import *
from .vgg import *
from .layers import *
from .nasnet import *
from .pnasnet import *
from .convnext import *
from .dpn import *
from .efficientnet import *
from .ghostnet import *
from .repvgg import *
from .swin_transformer import *
from .convit import *
