import clip
import numpy as np
import pytest
from PIL import Image

from mindspore.ops import softmax


@pytest.mark.parametrize("model_name", clip.available_models())
def test_consistency(model_name):
    device = "cpu"
    model, transform = clip.load(model_name, device=device)
    py_model, _ = clip.load(model_name, device=device)

    image = transform(Image.open("CLIP.png")).unsqueeze(0)
    text = clip.tokenize(["a diagram", "a dog", "a cat"])

    logits_per_image, _ = model(image, text)
    probs = softmax(logits_per_image, axis=-1).asnumpy()

    logits_per_image, _ = py_model(image, text)
    py_probs = softmax(logits_per_image, axis=-1).asnumpy()

    assert np.allclose(probs, py_probs, atol=0.01, rtol=0.1)
