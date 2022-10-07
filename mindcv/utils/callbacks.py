"""Callbacks for mindspore.Model"""
import os
#import stat
import numpy as np

from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore import save_checkpoint, SummaryRecord
from mindspore.train.callback import Callback
from mindspore.train._utils import _make_directory

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
                 best_ckpt_name="best.ckpt",
                 metric_name="accuracy",
                 dataset_sink_mode=True,
                 rank_id=None
                 ):
        super().__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.val_start_epoch = val_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.metric_name = metric_name
        self.best_res = 0
        self.val_interval = val_interval
        self.dataset_sink_mode = dataset_sink_mode
        self.summary_dir = summary_dir
        self.rank_id = rank_id if rank_id is not None else 0

        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def apply_eval(self):
        """Model evaluation, return validation accuracy."""
        return self.model.eval(self.dataset_val, dataset_sink_mode=self.dataset_sink_mode)

    def on_train_epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with highest validation accuracy.
        """
        callback_params = run_context.original_args()
        cur_epoch = callback_params.cur_epoch_num

        # record loss curve
        loss = self._get_loss(callback_params)
        self.summary_record.add_value('scalar', f'train_loss_{self.rank_id}', loss)

        # val while training if validation loader is not None
        if (self.dataset_val is not None) and (self.rank_id==0):
            if cur_epoch >= self.val_start_epoch and (cur_epoch - self.val_start_epoch) % self.val_interval == 0:
                # do validation
                res = self.apply_eval()
                print(f"Validation: epoch: {cur_epoch}, metrics: {res}")

                # record val acc
                val_acc = res[self.metric_name]
                if not isinstance(val_acc, Tensor):
                    val_acc = Tensor(val_acc)
                self.summary_record.add_value('scalar', 'val_' + self.metric_name, val_acc)

                # Save the best ckpt file
                if res[self.metric_name] > self.best_res:
                    self.best_res = res[self.metric_name]
                    if self.save_best_ckpt and (self.rank_id==0):
                        save_checkpoint(callback_params.train_network, self.best_ckpt_path, async_save=True)
                        print(f"Save the best {self.metric_name} ckpt, the {self.metric_name} is {self.best_res}")
                print("-" * 80)

        self.summary_record.record(callback_params.cur_step_num)

    # pylint: disable=unused-argument
    def on_train_end(self, run_context):
        if self.dataset_val is not None and self.rank_id==0:
            print(f"End of validation the best {self.metric_name} is: {self.best_res}, "
              f"save the best ckpt file in {self.best_ckpt_path}", flush=True)
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
        self._summary_dir = _make_directory(summary_dir,"summary_dir")
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
