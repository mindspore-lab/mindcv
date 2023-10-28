import hashlib
import os
import warnings
from typing import List, Union
from urllib import request

from PIL import Image
from pkg_resources import packaging
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, load_checkpoint
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import CenterCrop, Normalize, Resize, ToPIL, ToTensor

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from mindspore.dataset.vision import Inter

    BICUBIC = Inter.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if packaging.version.parse(ms.__version__) < packaging.version.parse("2.0.0"):
    warnings.warn("MindSpore version 2.0.0 or higher is recommended")

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50-5d39bdab.ckpt",
    "RN101": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN101-a9edcaa9.ckpt",
    "RN50x4": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x4-7b8cdb29.ckpt",
    "RN50x16": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x16-66ea7861.ckpt",
    "RN50x64": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x64-839951e0.ckpt",
    "ViT-B/32": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_32-34c32b89.ckpt",
    "ViT-B/16": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_16-99cbeeee.ckpt",
    "ViT-L/14": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_L_14-1d8bde7f.ckpt",
    "ViT-L/14@336px": "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_L_14_336px-9ed46dee.ckpt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-1].split("-")[-1].split(".")[0]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 not in hashlib.sha256(open(download_target, "rb").read()).hexdigest():
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            ToPIL(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], is_hwc=False),
        ]
    )


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: str = "Ascend", mode: int = 1, download_root: str = None):
    """Load a CLIP model and a set of transform operations to the image input.

    Parameters
    ----------
    name : str
        A model name or the path to a model checkpoint containing the parameter_dict, model names are listed by
        `clip.available_models()`.

    device : str
        The device to put the loaded model, must be one of CPU, GPU, Ascend

    mode : int
        GRAPH_MODE(0) or PYNATIVE_MODE(1).

    download_root: str
        Path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : mindspore.nn.Cell
        The CLIP model

    preprocess : Callable[[PIL.Image], mindspore.Tensor]
        A mindspore vision transform that converts a PIL image into a tensor that the returned model can
        take as its input.
    """
    ms.set_context(device_target=device, mode=mode)
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        ckp_dict = load_checkpoint(model_path)
    elif os.path.isfile(name):
        ckp_dict = load_checkpoint(name)
    else:
        raise ValueError(f"{name} not found; available models = {available_models()}")

    model = build_model(ckp_dict)
    if str(device).lower() == "cpu":
        model.to_float(ms.float32)
    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Tensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = ms.ops.zeros((len(all_tokens), context_length), dtype=ms.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = Tensor(tokens)

    return result
