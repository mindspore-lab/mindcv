"""Callbacks for mindspore.Model"""
import logging
import os
from time import time

import numpy as np

import mindspore as ms
from mindspore import ParameterTuple, Tensor, nn, ops
from mindspore.train import Callback, SummaryRecord, load_param_into_net, save_checkpoint

from .checkpoint_manager import CheckpointManager
from .reduce_manager import AllReduceSum

__all__ = [
    "StateMonitor",
    "ValCallback",
]

_logger = logging.getLogger(__name__)


class StateMonitor(Callback):
    """
    Train loss and validation accuracy monitor, after each epoch save the
    best checkpoint file with the highest validation accuracy.
    """

    def __init__(
        self,
        model,
        model_name="",
        model_ema=False,
        last_epoch=0,
        dataset_sink_mode=True,
        dataset_val=None,
        metric_name=("accuracy",),
        val_interval=1,
        val_start_epoch=1,
        save_best_ckpt=True,
        ckpt_save_dir="./",
        ckpt_save_interval=1,
        ckpt_save_policy=None,
        ckpt_keep_max=10,
        summary_dir="./",
        log_interval=100,
        rank_id=None,
        device_num=None,
    ):
        super().__init__()
        # model
        self.model = model
        self.model_name = model_name
        self.model_ema = model_ema
        self.last_epoch = last_epoch
        self.dataset_sink_mode = dataset_sink_mode
        # evaluation
        self.dataset_val = dataset_val
        self.metric_name = metric_name
        self.val_interval = val_interval
        self.val_start_epoch = val_start_epoch
        # logging
        self.best_res = 0
        self.best_epoch = -1
        self.save_best_ckpt = save_best_ckpt
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_interval = ckpt_save_interval
        self.ckpt_save_policy = ckpt_save_policy
        self.ckpt_keep_max = ckpt_keep_max
        self.ckpt_manager = CheckpointManager(ckpt_save_policy=self.ckpt_save_policy)
        self._need_flush_from_cache = True
        self.summary_dir = summary_dir
        self.log_interval = log_interval
        # system
        self.rank_id = rank_id if rank_id is not None else 0
        self.device_num = device_num if rank_id is not None else 1
        if self.rank_id in [0, None]:
            os.makedirs(ckpt_save_dir, exist_ok=True)
            self.log_file = os.path.join(ckpt_save_dir, "result.log")
            log_line = "".join(
                f"{s:<20}" for s in ["Epoch", "TrainLoss", *metric_name, "TrainTime", "EvalTime", "TotalTime"]
            )
            with open(self.log_file, "w", encoding="utf-8") as fp:  # writing the title of result.log
                fp.write(log_line + "\n")
        if self.device_num > 1:
            self.all_reduce = AllReduceSum()
        # timestamp
        self.step_ts = None
        self.epoch_ts = None
        self.step_time_accum = 0
        # model_ema
        if self.model_ema:
            self.hyper_map = ops.HyperMap()
            self.online_params = ParameterTuple(self.model.train_network.get_parameters())
            self.swap_params = self.online_params.clone("swap", "zeros")

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def apply_eval(self, run_context):
        """Model evaluation, return validation accuracy."""
        if self.model_ema:
            cb_params = run_context.original_args()
            self.hyper_map(ops.assign, self.swap_params, self.online_params)
            ema_dict = dict()
            net = self._get_network_from_cbp(cb_params)
            for param in net.get_parameters():
                if param.name.startswith("ema"):
                    new_name = param.name.split("ema.")[1]
                    ema_dict[new_name] = param.data
            load_param_into_net(self.model.train_network.network, ema_dict)
            res_dict = self.model.eval(self.dataset_val, dataset_sink_mode=False)
            self.hyper_map(ops.assign, self.online_params, self.swap_params)
        else:
            res_dict = self.model.eval(self.dataset_val, dataset_sink_mode=False)
        res_array = ms.Tensor(list(res_dict.values()), ms.float32)
        if self.device_num > 1:
            res_array = self.all_reduce(res_array)
            res_array /= self.device_num
        res_array = res_array.asnumpy()
        return res_array

    def on_train_step_begin(self, run_context):
        self.step_ts = time()

    def on_train_epoch_begin(self, run_context):
        self.epoch_ts = time()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        num_epochs = cb_params.epoch_num
        num_batches = cb_params.batch_num
        # num_steps = num_batches * num_epochs
        # cur_x start from 1, end at num_xs, range: [1, num_xs]
        cur_step = cb_params.cur_step_num + self.last_epoch * num_batches
        cur_epoch = cb_params.cur_epoch_num + self.last_epoch
        cur_batch = (cur_step - 1) % num_batches + 1

        self.step_time_accum += time() - self.step_ts
        if cur_batch % self.log_interval == 0 or cur_batch == num_batches or cur_batch == 1:
            lr = self._get_lr_from_cbp(cb_params)
            loss = self._get_loss_from_cbp(cb_params)
            _logger.info(
                f"Epoch: [{cur_epoch}/{num_epochs}], "
                f"batch: [{cur_batch}/{num_batches}], "
                f"loss: {loss.asnumpy():.6f}, "
                f"lr: {lr.asnumpy():.6f}, "
                f"time: {self.step_time_accum:.6f}s"
            )
            self.step_time_accum = 0

    def on_train_epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with the highest validation accuracy.
        """
        cb_params = run_context.original_args()
        num_epochs = cb_params.epoch_num
        num_batches = cb_params.batch_num
        cur_step = cb_params.cur_step_num + self.last_epoch * num_batches
        cur_epoch = cb_params.cur_epoch_num + self.last_epoch
        cur_batch = (cur_step - 1) % num_batches + 1

        train_time = time() - self.epoch_ts
        loss = self._get_loss_from_cbp(cb_params)

        val_time = 0
        res = np.zeros(len(self.metric_name), dtype=np.float32)
        # val while training if validation loader is not None
        if (
            self.dataset_val is not None
            and cur_epoch >= self.val_start_epoch
            and (cur_epoch - self.val_start_epoch) % self.val_interval == 0
        ):
            val_time = time()
            res = self.apply_eval(run_context)
            val_time = time() - val_time
            # record val acc
            metric_str = "Validation "
            for i in range(len(self.metric_name)):
                metric_str += f"{self.metric_name[i]}: {res[i]:.4%}, "
            metric_str += f"time: {val_time:.6f}s"
            _logger.info(metric_str)
            # save the best ckpt file
            if res[0] > self.best_res:
                self.best_res = res[0]
                self.best_epoch = cur_epoch
                _logger.info(f"=> New best val acc: {res[0]:.4%}")

        # save checkpoint
        if self.rank_id in [0, None]:
            if self.save_best_ckpt and self.best_epoch == cur_epoch:  # always save ckpt if cur epoch got best acc
                best_ckpt_save_path = os.path.join(self.ckpt_save_dir, f"{self.model_name}_best.ckpt")
                save_checkpoint(cb_params.train_network, best_ckpt_save_path, async_save=True)
            if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == num_epochs):
                if self._need_flush_from_cache:
                    self._flush_from_cache(cb_params)
                # save optim for resume
                optimizer = self._get_optimizer_from_cbp(cb_params)
                optim_save_path = os.path.join(self.ckpt_save_dir, f"optim_{self.model_name}.ckpt")
                save_checkpoint(optimizer, optim_save_path, async_save=True)
                # keep checkpoint files number equal max number.
                ckpt_save_path = os.path.join(self.ckpt_save_dir, f"{self.model_name}-{cur_epoch}_{cur_batch}.ckpt")
                _logger.info(f"Saving model to {ckpt_save_path}")
                self.ckpt_manager.save_ckpoint(
                    cb_params.train_network,
                    num_ckpt=self.ckpt_keep_max,
                    metric=res[0] if len(self.metric_name) > 0 else 0.0,
                    save_path=ckpt_save_path,
                )

        # logging
        total_time = time() - self.epoch_ts
        _logger.info(
            f"Total time since last epoch: {total_time:.6f}(train: {train_time:.6f}, val: {val_time:.6f})s, "
            f"ETA: {(num_epochs - cur_epoch) * total_time:.6f}s"
        )
        _logger.info("-" * 80)
        if self.rank_id in [0, None]:
            log_line = "".join(
                f"{s:<20}"
                for s in [
                    f"{cur_epoch}",
                    f"{loss.asnumpy():.6f}",
                    *[f"{i:.4%}" for i in res],
                    f"{train_time:.2f}",
                    f"{val_time:.2f}",
                    f"{total_time:.2f}",
                ]
            )
            with open(self.log_file, "a", encoding="utf-8") as fp:
                fp.write(log_line + "\n")

        # summary
        self.summary_record.add_value("scalar", f"train_loss_{self.rank_id}", loss)
        for i in range(len(res)):
            self.summary_record.add_value(
                "scalar", f"val_{self.metric_name[i]}_{self.rank_id}", Tensor(res[i], dtype=ms.float32)
            )
        self.summary_record.record(cur_step)

    def on_train_end(self, run_context):
        _logger.info("Finish training!")
        if self.dataset_val is not None:
            _logger.info(
                f"The best validation {self.metric_name[0]} is: {self.best_res:.4%} at epoch {self.best_epoch}."
            )
        _logger.info("=" * 80)

    def _get_network_from_cbp(self, cb_params):
        if self.dataset_sink_mode:
            network = cb_params.train_network.network
        else:
            network = cb_params.train_network
        return network

    def _get_optimizer_from_cbp(self, cb_params):
        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        elif self.dataset_sink_mode:
            optimizer = cb_params.train_network.network.optimizer
        else:
            optimizer = cb_params.train_network.optimizer
        return optimizer

    def _get_lr_from_cbp(self, cb_params):
        optimizer = self._get_optimizer_from_cbp(cb_params)
        if optimizer.global_step < 1:
            _logger.warning(
                "`global_step` of optimizer is less than 1. It seems to be a overflow at the first step. "
                "If you keep seeing this message, it means that the optimizer never actually called."
            )
            optim_step = Tensor((0,), ms.int32)
        else:  # if the optimizer is successfully called, the global_step will actually be the value of next step.
            optim_step = optimizer.global_step - 1
        if optimizer.dynamic_lr:
            if isinstance(optimizer.learning_rate, nn.CellList):
                # return the learning rates of the first parameter if dynamic_lr
                lr = optimizer.learning_rate[0](optim_step)[0]
            else:
                lr = optimizer.learning_rate(optim_step)[0]
        else:
            lr = optimizer.learning_rate
        return lr

    def _get_loss_from_cbp(self, cb_params):
        """
        Get loss from the network output.
        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        output = cb_params.net_outputs
        if output is None:
            _logger.warning("Can not find any output by this network, so SummaryCollector will not collect loss.")
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            _logger.warning(
                "The output type could not be identified, expect type is one of "
                "[int, float, Tensor, list, tuple], so no loss was recorded in SummaryCollector."
            )
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = Tensor(np.mean(loss.asnumpy()))
        return loss

    def _flush_from_cache(self, cb_params):
        """Flush cache data to host if tensor is cache enable."""
        has_cache_params = False
        params = cb_params.train_network.get_parameters()
        for param in params:
            if param.cache_enable:
                has_cache_params = True
                Tensor(param).flush_from_cache()
        if not has_cache_params:
            self._need_flush_from_cache = False


class ValCallback(Callback):
    def __init__(self, log_interval=100):
        super().__init__()
        self.log_interval = log_interval
        self.ts = time()

    def on_eval_step_end(self, run_context):
        cb_params = run_context.original_args()
        num_batches = cb_params.batch_num
        cur_step = cb_params.cur_step_num

        if cur_step % self.log_interval == 0 or cur_step == num_batches:
            print(f"batch: {cur_step}/{num_batches}, time: {time() - self.ts:.6f}s")
            self.ts = time()
