import os
import re
from enum import Enum

DATA_PRECISION = 2


class MatchWay(Enum):
    NUM = 0
    NUM_LAST_1 = 1  # get the last data match the need
    LEFT_RIGHT = 2  #
    FORMAT_DICT = 3  # json file
    RECALL = 4


def str_re_find(contents, left="", right="", key="", recall=None, match_way=MatchWay.NUM_LAST_1):
    """
        string match.
        support dictory,if contents like "{'test':"s,}"
        support left and right match
        default match last num
    """
    results = None
    if match_way == MatchWay.NUM:
        pattern = r"[1-9][0-9]*\.?[0-9]*|0\.\d+|[1-9][0-9]*\.?[0-9]*[Ee]?[+-]?\d+"
        results = re.findall(pattern, contents)
    elif match_way == MatchWay.NUM_LAST_1:
        results = re.findall(r"0\.\d+", contents)
    elif match_way == MatchWay.LEFT_RIGHT:
        # +- is science count way
        results = re.findall(r"{p_left}\s*([\'\"a-zA-Z0-9\._+-]+)\s*{p_right}"
                             .format(p_left=left, p_right=right), contents)
    elif match_way == MatchWay.FORMAT_DICT:
        pattern = r"[\'\"]{}[\'\"]\s*:\s*([\'\"a-zA-Z0-9\._]+)".format(key)
        results = re.findall(pattern, contents)
    elif match_way == MatchWay.RECALL:
        results = recall(contents)
    if results and isinstance(results, list) or isinstance(results, tuple):
        return results[-1]

    return results


def consult_evallog(eval_log, left="", right="", key="", recall=None, match_way=MatchWay.NUM_LAST_1):
    """
        consult the metric in eval_log
    """
    acc = None
    # eval_log
    if os.path.exists(eval_log):
        with open(eval_log, "r") as evalfile:
            contents = evalfile.read()

            acc = str_re_find(contents, left=left, right=right, key=key, recall=recall, match_way=match_way)
            try:
                if float(acc) < 1.0:
                    acc = float(acc) * 100
                acc = str(round(float(acc), DATA_PRECISION))
            finally:
                pass

    return acc


def find_datas_from_file(file_name, keys=None, output_type=str, past=False):
    """
    this function will checkt the first key match the maxNum
    :param fileName:
    :param keys:
    :param output_type:
    :return:
    """
    if past:
        return ["", ""]
    if keys is None:
        keys = [r",.*per\D*step\D*time\D*(\d+\.\d+)", r"ModelZoo-compile_time\D+(\d+:\d+:\d+\.\d+)"]

    results_list = []
    if os.path.exists(file_name) and keys:
        contents = []
        with open(file_name, "r") as trainfile:
            contents = trainfile.readlines()

        results_list = [list() for i in keys]

        for text in contents:
            for key in keys:
                result = re.search(key, text)
                if result:
                    results_list[keys.index(key)].append(result.group(1))

        if callable(output_type):
            for result in results_list:
                index = results_list.index(result)
                temp = list(map(output_type, results_list[index]))
                results_list[index] = temp

        if len(results_list) == 1:
            return results_list[0]
    else:
        raise Exception("file:{}\n[ERROR]not existed.please check the path and run again!".format(file_name))

    return results_list


def copy_net_config(test_config_path, source, copy_yaml_name):
    if not os.path.exists(test_config_path):
        os.makedirs(test_config_path)
    target = test_config_path + copy_yaml_name
    os.system(f"cp {source} {target}")


def modify_net_config(configfile, matchlist, replacelist, flags=re.DOTALL, check=True):
    """
        check the configinfo in the configfile,
        the rule of match configinfo is not exactly
    :param cofigfile: str
    :param caseSensitive:
    :param configInfo: e.g. batchsize=90,epoch=90,
    :return:
    """
    replacelist = list(replacelist)
    if os.path.exists(configfile) and configfile:
        with open(configfile, "r+") as cfg_file:
            contents_input = cfg_file.read()
            contents_output = None
            for key in replacelist:
                index = replacelist.index(key)
                match_str = matchlist[index]
                replace_str = replacelist[index]

                contents_output = re.sub(match_str, replace_str, contents_input, flags=flags)
                if contents_input == contents_output:
                    if replace_str not in contents_input and check:
                        raise IOError('FILE-{file}\n modify the config-[{conf}] Failed'
                                      .format(file=configfile, conf=match_str))
                else:
                    contents_input = contents_output

            if contents_output is None:
                return

            cfg_file.seek(0, 0)
            cfg_file.truncate()
            cfg_file.write(contents_output)
            cfg_file.flush()
    else:
        raise FileNotFoundError('FILE-{} not Exist!!'.format(configfile))

