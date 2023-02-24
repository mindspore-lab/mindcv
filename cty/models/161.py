import subprocess
import pytest
from PyTest.testcase.mindcv.utils.tools import copy_net_config, modify_net_config, find_datas_from_file, \
    consult_evallog, MatchWay
from PyTest.testcase.mindcv.utils.config import *

check_performance = True
check_acc = True
performance_std = 500000
Top_1 = 79.09
Top_5 = 94.66
mindcv_config_path = mindcv_path
data_path = dataset_path + "ImageNet_Original/"
mindcv_config_file = "configs/densenet/densenet_161_gpu.yaml"
test_config_path = "./test_config/"
net_name = "densenet161"
dataset_name_list = ["imagenet2012"]


def test_densenet_161_gpu_001():
    ckpt_path = hub_ckpt + net_name + "-120_5004_Ascend.ckpt"
    commom_log_part_name = net_name + dataset_name_list[0]
    validate_cmd = f"python {mindcv_config_path}validate.py -c {mindcv_config_path}{mindcv_config_file} --data_dir {data_path} --ckpt_path {ckpt_path}"
    print(f'Running command: \n{validate_cmd}')
    p = subprocess.Popen(validate_cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    if out:
        out = out.decode()
        val_out_log = commom_log_part_name + "gpu_val_out.log"
        with open(val_out_log, "w") as out_log:
            out_log.write(out)
    if err:
        err = err.decode()
        val_err_log = commom_log_part_name + "gpu_val_err.log"
        with open(val_err_log, "w") as err_log:
            err_log.write(err)

    if check_acc:
        Top_1_Accuracy = consult_evallog(val_out_log, key=r"Top_1_Accuracy", match_way=MatchWay.FORMAT_DICT)
        print('Val acc: ', Top_1_Accuracy)
        Top_5_Accuracy = consult_evallog(val_out_log, key=r"Top_5_Accuracy", match_way=MatchWay.FORMAT_DICT)
        print('Val acc: ', Top_5_Accuracy)
        assert float(Top_1_Accuracy) >= Top_1, 'Top_1 is too low'
        assert float(Top_5_Accuracy) >= Top_5, 'Top_5 is too low'


@pytest.mark.level('level0')
def test_densenet_161_gpu_002():
    common_name = "_gpu_train_8p_0_"
    combine_common_name = net_name + common_name + dataset_name_list[0]
    copy_yaml_name = combine_common_name + ".yaml"
    copy_net_config(test_config_path, mindcv_config_path + mindcv_config_file, copy_yaml_name)
    common_yaml_path_name = test_config_path + copy_yaml_name
    modify_net_config(common_yaml_path_name, [r"epoch_size: 120"], [r"epoch_size: 5"])
    modify_net_config(common_yaml_path_name, [r"./ckpt"], [f"./{combine_common_name}_ckpt"])
    train_cmd = f"mpirun --allow-run-as-root -n 8 python {mindcv_config_path}train.py --config {common_yaml_path_name} --data_dir {data_path}"
    print(f'Running command: \n{train_cmd}')
    p = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    if out:
        out = out.decode()
        train_out_log = combine_common_name + "_out.log"
        with open(train_out_log, "w") as out_log:
            out_log.write(out)
    if err:
        err = err.decode()
        train_err_log = combine_common_name + "_err.log"
        with open(train_err_log, "w") as err_log:
            err_log.write(err)

    if check_performance:
        performance, compile_time = find_datas_from_file(train_out_log,
                                                         keys=[r"last epoch:\D*(\d+\.\d+)",
                                                               r"ModelZoo-compile_time\D+(\d+:\d+:\d+\.\d+)"],
                                                         past=False)
        assert float(performance[-1]) <= performance_std, 'performance is too slow'


@pytest.mark.level('level0')
def test_densenet_161_gpu_003():
    common_name = "_gpu_train_8p_1_"
    combine_common_name = net_name + common_name + dataset_name_list[0]
    copy_yaml_name = combine_common_name + ".yaml"
    copy_net_config(test_config_path, mindcv_config_path + mindcv_config_file, copy_yaml_name)
    common_yaml_path_name = test_config_path + copy_yaml_name
    modify_net_config(common_yaml_path_name, [r"epoch_size: 120"], [r"epoch_size: 5"])
    modify_net_config(common_yaml_path_name, [r"./ckpt"], [f"./{combine_common_name}_ckpt"])
    train_cmd = f"mpirun --allow-run-as-root -n 8 python {mindcv_config_path}train.py --config {common_yaml_path_name} --data_dir {data_path} --mode 1"
    print(f'Running command: \n{train_cmd}')
    p = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    if out:
        out = out.decode()
        train_out_log = combine_common_name + "_out.log"
        with open(train_out_log, "w") as out_log:
            out_log.write(out)
    if err:
        err = err.decode()
        train_err_log = combine_common_name + "_err.log"
        with open(train_err_log, "w") as err_log:
            err_log.write(err)

    if check_performance:
        performance, compile_time = find_datas_from_file(train_out_log,
                                                         keys=[r"last epoch:\D*(\d+\.\d+)",
                                                               r"ModelZoo-compile_time\D+(\d+:\d+:\d+\.\d+)"],
                                                         past=False)
        assert float(performance[-1]) <= performance_std, 'performance is too slow'


@pytest.mark.level('level0')
def test_densenet_161_gpu_004():
    common_name = "_gpu_train_1p_0_"
    combine_common_name = net_name + common_name + dataset_name_list[0]
    copy_yaml_name = combine_common_name + ".yaml"
    copy_net_config(test_config_path, mindcv_config_path + mindcv_config_file, copy_yaml_name)
    common_yaml_path_name = test_config_path + copy_yaml_name
    modify_net_config(common_yaml_path_name, [r"epoch_size: 120"], [r"epoch_size: 1"])
    modify_net_config(common_yaml_path_name, [r"./ckpt"], [f"./{combine_common_name}_ckpt"])
    train_cmd = f"python {mindcv_config_path}train.py --config {common_yaml_path_name} --data_dir {data_path} --distribute False"
    print(f'Running command: \n{train_cmd}')
    p = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    if out:
        out = out.decode()
        train_out_log = combine_common_name + "_out.log"
        with open(train_out_log, "w") as out_log:
            out_log.write(out)
    if err:
        err = err.decode()
        train_err_log = combine_common_name + "_err.log"
        with open(train_err_log, "w") as err_log:
            err_log.write(err)

    if check_performance:
        performance, compile_time = find_datas_from_file(train_out_log,
                                                         keys=[r"last epoch:\D*(\d+\.\d+)",
                                                               r"ModelZoo-compile_time\D+(\d+:\d+:\d+\.\d+)"],
                                                         past=False)
        assert float(performance[-1]) <= performance_std, 'performance is too slow'


@pytest.mark.level('level0')
def test_densenet_161_gpu_005():
    common_name = "_gpu_train_1p_1_"
    combine_common_name = net_name + common_name + dataset_name_list[0]
    copy_yaml_name = combine_common_name + ".yaml"
    copy_net_config(test_config_path, mindcv_config_path + mindcv_config_file, copy_yaml_name)
    common_yaml_path_name = test_config_path + copy_yaml_name
    modify_net_config(common_yaml_path_name, [r"epoch_size: 120"], [r"epoch_size: 1"])
    modify_net_config(common_yaml_path_name, [r"./ckpt"], [f"./{combine_common_name}_ckpt"])
    train_cmd = f"python {mindcv_config_path}train.py --config {common_yaml_path_name} --data_dir {data_path} --distribute False --mode 1"
    print(f'Running command: \n{train_cmd}')
    p = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    if out:
        out = out.decode()
        train_out_log = combine_common_name + "_out.log"
        with open(train_out_log, "w") as out_log:
            out_log.write(out)
    if err:
        err = err.decode()
        train_err_log = combine_common_name + "_err.log"
        with open(train_err_log, "w") as err_log:
            err_log.write(err)

    if check_performance:
        performance, compile_time = find_datas_from_file(train_out_log,
                                                         keys=[r"last epoch:\D*(\d+\.\d+)",
                                                               r"ModelZoo-compile_time\D+(\d+:\d+:\d+\.\d+)"],
                                                         past=False)
        assert float(performance[-1]) <= performance_std, 'performance is too slow'
