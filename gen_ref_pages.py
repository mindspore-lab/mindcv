"""Generate the code reference pages of models."""
import os
import sys

sys.path.append(".")

import importlib
import logging
from pathlib import Path

_logger = logging.getLogger('mkdocs')
_langs = ["en", "zh"]


def _gen_page(lang):
    full_doc_path = Path(f"docs/{lang}/reference/models.md")
    _logger.info(f"Generating reference page: {full_doc_path}")
    with open(full_doc_path, "w") as fd:
        print("# Models", file=fd)
        print("\n\n## Create Model", file=fd)
        print("\n### ::: mindcv.models.model_factory.create_model", file=fd)

        for path in sorted(Path("mindcv/models").rglob("*.py")):
            module_path = path.with_suffix("")  # eg: mindcv/models/resnet
            parts = list(module_path.parts)  # eg: ["mindcv", "models", "resnet"]
            if parts[-1].startswith("__") or parts[-2] == "layers":
                continue
            # fileter out utility modules
            if parts[-1] in ["model_factory", "registry", "utils", "helpers"]:
                continue
            # filter out the net module which is replaced by the net function with the same name
            # TODO: we need to change mechanism of model importing
            if parts[-1] in ["googlenet", "inception_v3", "inception_v4", "xception", "pnasnet"]:
                continue

            try:
                print(f"\n\n## {parts[-1]}", file=fd)
                identifier = ".".join(parts)  # eg: mindcv.models.resnet
                mod = importlib.import_module(identifier)
                for mem in sorted(set(mod.__all__)):
                    print(f"\n### ::: {identifier}.{mem}", file=fd)
            except Exception as err:
                _logger.warning(f"Cannot generate reference of {identifier}, error: {err}.")


def _del_page(lang):
    full_doc_path = Path(f"docs/{lang}/reference/models.md")
    _logger.info(f"Cleaning generated reference page: {full_doc_path}")
    os.remove(full_doc_path)


def on_startup(command, dirty):
    for lang in _langs:
        _gen_page(lang)


def on_shutdown():
    for lang in _langs:
        _del_page(lang)
