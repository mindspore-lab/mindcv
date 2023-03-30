import os
import sys

sys.path.append(".")

import pytest
import yaml

from config import _check_cfgs_in_parser, create_parser, parse_args


def test_checker_valid():
    cfgs = yaml.safe_load(
        """
        mode: 1
        dataset: imagenet
        """
    )
    _, parser = create_parser()
    _check_cfgs_in_parser(cfgs, parser)


def test_checker_invalid():
    cfgs = yaml.safe_load(
        """
        mode: 1
        dataset: imagenet
        valid: False
        """
    )
    _, parser = create_parser()
    with pytest.raises(KeyError) as exc_info:
        _check_cfgs_in_parser(cfgs, parser)
    assert exc_info.type is KeyError
    assert exc_info.value.args[0] == "valid does not exist in ArgumentParser!"


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dataset", ["mnist", "imagenet"])
def test_parse_args_without_yaml(mode, dataset):
    args = parse_args([f"--mode={mode}", f"--dataset={dataset}"])
    assert args.mode == mode
    assert args.dataset == dataset
    assert args.amp_level == "O0"  # default value


@pytest.mark.parametrize("cfg_yaml", ["configs/resnet/resnet_18_ascend.yaml"])
@pytest.mark.parametrize("mode", [1])
@pytest.mark.parametrize("dataset", ["mnist"])
def test_parse_args_with_yaml(cfg_yaml, mode, dataset):
    args = parse_args([f"--config={cfg_yaml}", f"--mode={mode}", f"--dataset={dataset}"])
    assert args.mode == mode
    assert args.dataset == dataset
    with open(cfg_yaml, "r") as f:
        cfg = yaml.safe_load(f)
        model = cfg["model"]
    assert args.model == model  # from cfg.yaml


def test_parse_args_from_all_yaml():
    cfgs_root = "configs"
    cfg_paths = []
    for dirpath, dirnames, filenames in os.walk(cfgs_root):
        for filename in filenames:
            if filename.endswith((".yaml", "yml")):
                cfg_paths.append(os.path.join(dirpath, filename))
    for cfg_yaml in cfg_paths:
        try:
            args = parse_args([f"--config={cfg_yaml}"])
            with open(cfg_yaml, "r") as f:
                cfg = yaml.safe_load(f)
                model = cfg["model"]
            assert args.model == model
        except KeyError as e:
            raise AssertionError(f"{cfg_yaml} has some invalid options: {e}")
