import numpy as np

from mindspore.train.metrics.metric import _check_onehot_data, rearrange_inputs
from mindspore.train.metrics.topk import TopKCategoricalAccuracy


class TopKCategoricalAccuracyForTokenData(TopKCategoricalAccuracy):
    @rearrange_inputs
    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError(
                "For 'TopKCategoricalAccuracy.update', "
                "it needs 2 inputs (predicted value, true value), "
                "but got 'inputs' size: {}.".format(len(inputs))
            )
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        y_pred = np.reshape(y_pred, (-1, y_pred.shape[-1]))
        y = np.reshape(y, (-1,))

        inds = y >= 0  # remove the padded token data
        y_pred = y_pred[inds]
        y = y[inds]

        if y_pred.ndim == y.ndim and _check_onehot_data(y):
            y = y.argmax(axis=1)
        indices = np.argsort(-y_pred, axis=1)[:, : self.k]
        repeated_y = y.reshape(-1, 1).repeat(self.k, axis=1)
        correct = np.equal(indices, repeated_y).sum(axis=1)
        self._correct_num += correct.sum()
        self._samples_num += repeated_y.shape[0]


class Top1CategoricalAccuracyForTokenData(TopKCategoricalAccuracyForTokenData):
    def __init__(self):
        super(Top1CategoricalAccuracyForTokenData, self).__init__(1)


class Top5CategoricalAccuracyForTokenData(TopKCategoricalAccuracyForTokenData):
    def __init__(self):
        super(Top5CategoricalAccuracyForTokenData, self).__init__(5)
