"""config"""
dataset_path = "/data1/mindspore_dataset/"
mindcv_path = "/media/nvme1n1/cty/test_mind/mindcv/"
hub_ckpt = "/media/nvme1n1/cty/test_mind/mindcv/hub_ckpt/"

# Ascend hccl commuicate
rank_ip = "/data1/mindspore_dataset/rank_table/r07/hccl_8p.json"
rank_ip_4p = "/data1/mindspore_dataset/rank_table_4p.json"
rank_ip_16p = "/data1/mindspore_dataset/rank_table/r07/hccl_16p.json"

# groups config
mpi_hostsfile = "/data1/mindspore_dataset/mpihostsfile"  # GPU
groups_conig = "/data1/mindspore_dataset/group_settings.json"  # Ascend && gpu
code_dir = "../models"
output_dir = "../../performances"
geir_dir = "../../geir"
mindir_dir = "../../mindir"
backup_data_dir = "../../backup"
MAX_CKPT_NUM = 5

past_ckpt_dir = "/../../ckpt_hub"
past_performance_dir = "../../past_performances"
err_recall_only_eval = "err_recall_only_eval.yaml"

# Data
DATA_PRECISION = 2
