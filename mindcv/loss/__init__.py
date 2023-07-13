""" loss init """
from . import asymmetric, binary_cross_entropy_smooth, cross_entropy_smooth, jsd, loss_factory
from .asymmetric import AsymmetricLossMultilabel, AsymmetricLossSingleLabel
from .binary_cross_entropy_smooth import BinaryCrossEntropySmooth
from .cross_entropy_smooth import CrossEntropySmooth
from .jsd import JSDCrossEntropy
from .loss_factory import create_loss

__all__ = []
__all__.extend(loss_factory.__all__)
