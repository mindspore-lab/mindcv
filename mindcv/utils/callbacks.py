"""Callbacks for mindspore.Model"""
import os
from time import time
#import stat
import numpy as np

import mindspore as ms
from mindspore import log as logger
from mindspore import Tensor, save_checkpoint, SummaryRecord
from mindspore.train.callback import Callback
from mindspore.train._utils import _make_directory

from .checkpoint_manager import CheckpointManager
from .reduce_manager import Allreduce

class StateMonitor(Callback):
    """
    Train loss and validation accuracy monitor, after each epoch save the
    best checkpoint file with highest validation accuracy.
    """
    def __init__(self,
                 model,
                 summary_dir="./",
                 dataset_val=None,
                 val_interval=1,
                 val_start_epoch=1,
                 save_best_ckpt=True,
                 ckpt_dir="./",
                 ckpt_save_interval=1,
                 best_ckpt_name="best.ckpt",
                 metric_name="accuracy",
                 rank_id=None,
                 device_num=None,
                 log_interval=100,
                 model_name='',
                 last_epoch=0,
                 keep_checkpoint_max=10,
                 ckpt_save_policy=None
                 ):
        super().__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.val_start_epoch = val_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.metric_name = metric_name
        self.best_res = 0
        self.val_interval = val_interval
        self.summary_dir = summary_dir
        self.rank_id = rank_id if rank_id is not None else 0
        self.device_num = device_num if rank_id is not None else 1
        self.log_interval = log_interval
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.ckpt_save_interval = ckpt_save_interval
        self.last_epoch = last_epoch
        self.best_epoch= -1

        self.keep_checkpoint_max = keep_checkpoint_max
        self.ckpt_save_policy = ckpt_save_policy
        self._manager = CheckpointManager(ckpt_save_policy=self.ckpt_save_policy)
        self._need_flush_from_cache = True

        if self.rank_id in [0, None]:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            self.log_txt_fp = os.path.join(ckpt_dir, 'result.log')

            with open(self.log_txt_fp, 'w', encoding="utf-8") as fp:
                fp.write('Epoch\tTrainLoss\tValAcc\tTime\n')

            self.best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)

        if self.device_num > 1:
            self.all_reduce = Allreduce()

        self.start = time()
        self.epoch_start = time()

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def apply_eval(self):
        """Model evaluation, return validation accuracy."""
        return self.model.eval(self.dataset_val, dataset_sink_mode=False)

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        num_batches = cb_params.batch_num
        #global_step = cb_params.optimizer.global_step.asnumpy()[0]
        cur_epoch = cb_params.cur_epoch_num + self.last_epoch -1 #(global_step-1) // num_batches
        #cur_step_in_epoch = (global_step- 1) % cb_params.batch_num
        cur_step_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num)


        if (cur_step_in_epoch  + 1) % self.log_interval == 0 or \
                (cur_step_in_epoch  + 1) >= num_batches or cur_step_in_epoch == 0:
            step = cb_params.optimizer.global_step
            if cb_params.optimizer.dynamic_lr:
                cur_lr = cb_params.optimizer.learning_rate(step-1)[0].asnumpy()
            else:
                cur_lr = cb_params.optimizer.learning_rate.asnumpy()
            loss = self._get_loss(cb_params)

            print(f"Epoch: {cur_epoch+1}, "
                  f"batch:[{cur_step_in_epoch+1}/{num_batches}], "
                  f"loss:{loss.asnumpy():.6f}, lr: {cur_lr:.7f},  time:{time() - self.start:.6f}s")
            self.start = time()

    def on_train_epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with highest validation accuracy.
        """
        cb_params = run_context.original_args()
        # the global step may larger than batch_size * epoch due to graph mode async
        global_step = cb_params.optimizer.global_step.asnumpy()[0]
        cur_epoch = cb_params.cur_epoch_num + self.last_epoch
        cur_step_in_epoch = cb_params.batch_num #(global_step - 1) % cb_params.batch_num

        loss = self._get_loss(cb_params)
        self.summary_record.add_value('scalar', f'train_loss_{self.rank_id}', loss)

        # val while training if validation loader is not None
        val_acc_val = Tensor(-1.0)
        if self.dataset_val is not None:
            if cur_epoch >= self.val_start_epoch and (cur_epoch - self.val_start_epoch) % self.val_interval == 0:
                val_time = time()
                res = self.apply_eval()[self.metric_name]
                if self.device_num > 1:
                    res = self.all_reduce(Tensor(res, ms.float32)).asnumpy()
                    res /= self.device_num
                val_acc_val = 100 * res
                # record val acc
                if self.rank_id in [0, None]:
                    print(f"Validation {self.metric_name}: {val_acc_val:.3f}, time:{time() - val_time:.6f}s")
                    # Save the best ckpt file
                    if val_acc_val > self.best_res:
                        self.best_res = val_acc_val
                        self.best_epoch = cur_epoch
                        if self.save_best_ckpt and (self.rank_id == 0):
                            save_checkpoint(cb_params.train_network, self.best_ckpt_path, async_save=True)
                            print(f"=> New best val acc: {val_acc_val:.3f}")

                    if not isinstance(val_acc_val, Tensor):
                        val_acc_val = Tensor(val_acc_val)
                    self.summary_record.add_value('scalar', 'val_' + self.metric_name, val_acc_val)

        # log
        if self.rank_id in [0, None]:
            if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == cb_params.epoch_num):
                if self._need_flush_from_cache:
                    self._flush_from_cache(cb_params)

                # save optim for resume
                optim_save_path = os.path.join(self.ckpt_dir, f'optim_{self.model_name}.ckpt')
                ms.save_checkpoint(cb_params.optimizer, optim_save_path, async_save=True)

                cur_ckpoint_file = self.model_name + "-" + str(cur_epoch) + "_" \
                                   + str(cur_step_in_epoch) + ".ckpt"

                # keep checkpoint files number equal max number.
                ckpt_save_path = os.path.join(self.ckpt_dir, cur_ckpoint_file)
                ckpoint_filelist = self._manager.save_ckpoint(cb_params.train_network,
                                                              num_ckpt=self.keep_checkpoint_max,
                                                              metric=val_acc_val,
                                                              save_path=ckpt_save_path)
                if self.ckpt_save_policy == 'top_k':
                    print("Top K accuracy checkpoints:")
                    print('\n'.join(ckpt for ckpt,_ in ckpoint_filelist))
                else:
                    print(f"Saving model to {ckpt_save_path}")

            epoch_time = time() - self.epoch_start
            print(f'Total time since last epoch: {epoch_time:.3f}')
            print("-" * 80)
            self.epoch_start = time()

            with open(self.log_txt_fp, 'a', encoding="utf-8") as fp:
                fp.write(f'{cur_epoch}\t{loss.asnumpy():.7f}\t{val_acc_val.asnumpy():.3f}\t{epoch_time:.2f}\n')

        self.summary_record.record(int(global_step))

    # pylint: disable=unused-argument
    def on_train_end(self, run_context):
        if self.dataset_val is not None and self.rank_id == 0:
            print("Finish training!")
            print(f"The best validation {self.metric_name} is: {self.best_res} at epoch {self.best_epoch}.")
        print("=" * 80)

    def _get_loss(self, cb_params):
        """
        Get loss from the network output.
        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        output = cb_params.net_outputs
        if output is None:
            logger.warning("Can not find any output by this network, so SummaryCollector will not collect loss.")
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            logger.warning("The output type could not be identified, expect type is one of "
                           "[int, float, Tensor, list, tuple], so no loss was recorded in SummaryCollector.")
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

    def remove_oldest_ckpoint_file(self):
        """Remove the oldest checkpoint file from this checkpoint manager and also from the directory."""
        ckpoint_files = sorted(self._ckpoint_filelist, key=os.path.getmtime)
        self.remove_ckpoint_file(ckpoint_files[0])

class LossAccSummary(Callback):
    ''' A callback for recording loss and acc during training '''
    def __init__(self,
                 summary_dir,
                 model,
                 dataset_val,
                 val_interval=1,
                 val_start_epoch=1,
                 metric_name="accuracy"):
        super().__init__()
        self._summary_dir = _make_directory(summary_dir, "summary_dir")
        self.model = model
        self.dataset_val = dataset_val
        self.val_start_epoch = val_start_epoch
        self.metric_name = metric_name
        self.val_interval = val_interval

    def __enter__(self):
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = self._get_loss(cb_params)
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.val_start_epoch and (cur_epoch - self.val_start_epoch) % self.val_interval == 0:
            val_acc = self.model.eval(self.dataset_val)[self.metric_name]
            if not isinstance(val_acc, Tensor):
                val_acc = Tensor(val_acc)
            self.summary_record.add_value('scalar', 'test_dataset_' + self.metric_name, val_acc)

        self.summary_record.add_value('scalar', 'loss/auto', loss)
        self.summary_record.record(cb_params.cur_step_num)

    def _get_loss(self, cb_params):
        """
        Get loss from the network output.
        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        output = cb_params.net_outputs
        if output is None:
            logger.warning("Can not find any output by this network, so SummaryCollector will not collect loss.")
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            logger.warning("The output type could not be identified, expect type is one of "
                           "[int, float, Tensor, list, tuple], so no loss was recorded in SummaryCollector.")
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = Tensor(np.mean(loss.asnumpy()))
        return loss

class ValCallback(Callback):
    def __init__(self, log_step_interval=100):
        super().__init__()
        self.log_step_interval = log_step_interval

    def on_eval_step_end(self, run_context):
        cb_params = run_context.original_args()
        #cur_step_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num)
        if cb_params.cur_step_num % self.log_step_interval == 0:
            print(f'{cb_params.cur_step_num }/{cb_params.batch_num}')
