from functools import partial
from itertools import islice
from typing import Callable, Optional, Sequence, Union

from mindspore import ops


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_classifier(
    model,
    tokenizer,
    classnames: Sequence[str],
    templates: Sequence[Union[Callable, str]],
    num_classes_per_batch: Optional[int] = 10,
    use_tqdm: bool = False,
):
    """Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm

        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts)
        class_embeddings = ops.L2Normalize(-1)(model.encode_text(texts))
        class_embeddings = class_embeddings.reshape((num_batch_classes, num_templates, -1)).mean(axis=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    if num_classes_per_batch:
        batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
        zeroshot_weights = ops.cat(batched_embeds, axis=1)
    else:
        zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


def build_zero_shot_classifier_legacy(
    model,
    tokenizer,
    classnames: Sequence[str],
    templates: Sequence[Union[Callable, str]],
    use_tqdm: bool = False,
):
    """Build zero-shot classifier weights by iterating over class names 1 by 1
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    if use_tqdm:
        import tqdm

        iter_wrap = tqdm.tqdm
    else:
        iter_wrap = iter

    use_format = isinstance(templates[0], str)

    zeroshot_weights = []
    for classname in iter_wrap(classnames):
        texts = [template.format(classname) if use_format else template(classname) for template in templates]
        texts = tokenizer(texts)  # tokenize
        class_embeddings = model.encode_text(texts)
        class_embedding = ops.L2Normalize(-1)(class_embeddings).mean(axis=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = ops.stack(zeroshot_weights, axis=1)

    return zeroshot_weights
