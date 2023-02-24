import subprocess
import pytest
from PyTest.testcase.mindcv.utils.tools import copy_net_config, modify_net_config, find_datas_from_file, \
    consult_evallog, MatchWay
from PyTest.testcase.mindcv.utils.config import *


class validate_factory_models:
    def __int__(self, mindcv_config_path, mindcv_config_file, data_path, ckpt_path):
        self.mindcv_config_path = mindcv_config_path
        self.mindcv_config_file = mindcv_config_file
        self.data_path = data_path
        self.ckpt_path = ckpt_path


class train_factory_models():
    pass


class train_with_func_factory_models():
    pass


class validate_with_func_factory_models():
    pass


class infer_factory_models():
    pass
