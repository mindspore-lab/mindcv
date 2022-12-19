''' auto mixed precision related functions '''
import mindspore as ms
from mindspore.amp import LossScaler

class NoLossScaler(LossScaler):
    """
    No LossScaler
    """

    def __init__(self):
        super().__init__(1)

    def scale(self, inputs):
        return inputs

    def unscale(self, inputs):
        return inputs

    def adjust(self, grads_finite):
        return True


