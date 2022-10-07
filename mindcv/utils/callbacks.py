"""Callbacks for mindspore.Model"""
import os
import stat
import numpy as np

from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore import save_checkpoint, SummaryRecord
from mindspore.train.callback import Callback
from mindspore.train._utils import _make_directory


class ValAccSaveMonitor(Callback):
    """
    Train loss and validation accuracy monitor, after each epoch save the
    best checkpoint file with highest validation accuracy.
    Usage:
        >>> monitor = ValAccSaveMonitor(model, dataset_val, ckpt_dir='./model_ckpt')
    """

    def __init__(self,
                 model,
                 dataset_val,
                 interval=1,
                 eval_start_epoch=1,
                 save_best_ckpt=True,
                 ckpt_dir="./",
                 best_ckpt_name="best.ckpt",
                 metric_name="accuracy",
                 dataset_sink_mode=True):
        super().__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.eval_start_epoch = eval_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.metric_name = metric_name
        self.best_res = 0
        self.interval = interval
        self.dataset_sink_mode = dataset_sink_mode

        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.best_ckpt_path = os.path.join(ckpt_dir, best_ckpt_name)

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

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            # Validation result
            res = self.apply_eval()
            print(f"Eval result: epoch: {cur_epoch}, metrics: {res}")

            def remove_ckpt_file(file_name):
                os.chmod(file_name, stat.S_IWRITE)
                os.remove(file_name)

            # Save the best ckpt file
            if res[self.metric_name] >= self.best_res:
                self.best_res = res[self.metric_name]
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        remove_ckpt_file(self.best_ckpt_path)
                    save_checkpoint(callback_params.train_network, self.best_ckpt_path, async_save=True)
                    print(f"Save the best {self.metric_name} ckpt, the {self.metric_name} is {self.best_res}")
            print("-" * 80)

    # pylint: disable=unused-argument
    def on_train_end(self, run_context):
        print(f"End of validation the best {self.metric_name} is: {self.best_res}, "
              f"save the best ckpt file in {self.best_ckpt_path}", flush=True)
        print("=" * 80)

class LossAccSummary(Callback):
    def __init__(self,
                 summary_dir,
                 model,
                 dataset_val,
                 interval=1,
                 eval_start_epoch=1,
                 metric_name="accuracy"):
        super(LossAccSummary, self).__init__()
        self._summary_dir = _make_directory(summary_dir,"summary_dir")
        self.model = model
        self.dataset_val = dataset_val
        self.eval_start_epoch = eval_start_epoch
        self.metric_name = metric_name
        self.interval = interval

    def __enter__(self):
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = self._get_loss(cb_params)
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            test_acc = self.model.eval(self.dataset_val)[self.metric_name]
            if not isinstance(test_acc, Tensor):
                test_acc = Tensor(test_acc)
            self.summary_record.add_value('scalar', 'test_dataset_' + self.metric_name, test_acc)
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
