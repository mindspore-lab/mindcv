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

from .registry import is_model, model_entrypoint
from mindspore import load_checkpoint, load_param_into_net


def create_model(
        model_name: str,
        num_classes: int = 1000,
        pretrained=False,
        in_channels: int = 3,
        checkpoint_path: str = '',
        **kwargs):

    if checkpoint_path is not '' and pretrained:
        raise ValueError('checkpoint_path is mutually exclusive with pretrained')

    model_args = dict(num_classes=num_classes, pretrained=pretrained, in_channels=in_channels)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    model = create_fn(**model_args, **kwargs)

    if checkpoint_path:
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(model, param_dict)

    return model
