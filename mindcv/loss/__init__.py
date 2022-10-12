''' loss init '''
from .cross_entropy_smooth import CrossEntropySmooth
from .binary_cross_entropy_smooth import BinaryCrossEntropySmooth
from .loss_factory import create_loss

__all__ = []
__all__.extend(loss_factory.__all__)
