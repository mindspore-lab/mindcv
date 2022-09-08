#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""cross entropy smooth"""

import mindspore.nn as nn
import mindspore.ops as ops


class CrossEntropySmooth(nn.LossBase):

    def __init__(self, smooth_factor=0., factor=0.):
        super(CrossEntropySmooth, self).__init__()
        self.smoothing = smooth_factor
        self.confidence = 1. - smooth_factor
        self.factor = factor
        self.log_softmax = ops.LogSoftmax()
        self.gather = ops.Gather()
        self.expand = ops.ExpandDims()

    def construct(self, logit, label):
        loss_aux = 0
        if self.factor > 0:
            logit, aux = logit
            auxprobs = self.log_softmax(aux)
            nll_loss_aux = ops.gather_d((-1 * auxprobs), 1, self.expand(label, -1))
            nll_loss_aux = nll_loss_aux.squeeze(1)
            smooth_loss = -auxprobs.mean(axis=-1)
            loss_aux = (self.confidence * nll_loss_aux + self.smoothing * smooth_loss).mean()
        logprobs = self.log_softmax(logit)
        nll_loss_logit = ops.gather_d((-1 * logprobs), 1, self.expand(label, -1))
        nll_loss_logit = nll_loss_logit.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss_logit = (self.confidence * nll_loss_logit + self.smoothing * smooth_loss).mean()
        loss = loss_logit + self.factor * loss_aux
        return loss
