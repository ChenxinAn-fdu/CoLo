import os
import torch
import sys
from torch import nn

from fastNLP.core.callback import Callback
from fastNLP.core.utils import _get_model_device
from fastNLP import logger


class LrCallback(Callback):
    def __init__(self, args):
        super(LrCallback, self).__init__()
        self.args = args
        self.real_step = 0

    def on_step_end(self):
        if self.step % self.update_every == 0 and self.step > 0:
            self.real_step += 1
            cur_lr = self.args.max_lr * 100 * min(self.real_step ** (-0.5),
                                                  self.real_step * self.args.lr_warmup_steps ** (-1.5))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            if self.real_step % 1000 == 0:
                self.pbar.write('Current learning rate is {:.8f}, real_step: {}'.format(cur_lr, self.real_step))

    def on_epoch_end(self):
        self.pbar.write('Epoch {} is done !!!'.format(self.epoch))


def _save_model(model, model_name, save_dir, only_param=False):
    model_path = os.path.join(save_dir, model_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, nn.DataParallel):
        model = model.module
    if only_param:
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, model_path)
        logger.info(f"save checkpoints to ----> {model_path}  ")

    else:
        _model_device = _get_model_device(model)
        model.cpu()
        torch.save(model, model_path)
        model.to(_model_device)


class SaveModelCallback(Callback):
    def __init__(self, save_dir, top=3, warmup_ckpt=None, only_param=False, save_on_exception=False):
        super().__init__()

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        if top < 0:
            self.top = sys.maxsize
        else:
            self.top = top
        self._ordered_save_models = []
        self.warmup_ckpt = warmup_ckpt
        self.only_param = only_param
        self.save_on_exception = save_on_exception

    def on_train_begin(self):
        if not self.warmup_ckpt:
            self.save_dir = os.path.join(self.save_dir, self.trainer.start_time)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        metric_value = list(eval_result.values())[0][metric_key]
        self._save_this_model(metric_value)

    def _insert_into_ordered_save_models(self, pair):
        # pair:(metric_value, model_name)
        index = -1
        for _pair in self._ordered_save_models:
            if _pair[0] >= pair[0] and self.trainer.increase_better:
                break
            if not self.trainer.increase_better and _pair[0] <= pair[0]:
                break
            index += 1
        save_pair = None
        if len(self._ordered_save_models) < self.top or (len(self._ordered_save_models) >= self.top and index != -1):
            save_pair = pair
            self._ordered_save_models.insert(index + 1, pair)
        delete_pair = None
        if len(self._ordered_save_models) > self.top:
            delete_pair = self._ordered_save_models.pop(0)
        return save_pair, delete_pair

    def _save_this_model(self, metric_value):
        if self.warmup_ckpt:
            name = self.warmup_ckpt
        else:
            name = "epoch-{}_step-{}_{}-{:.6f}.pt".format(self.epoch, self.step, self.trainer.metric_key, metric_value)

        save_pair, delete_pair = self._insert_into_ordered_save_models((metric_value, name))
        if delete_pair:
            try:
                delete_model_path = os.path.join(self.save_dir, delete_pair[1])
                if os.path.exists(delete_model_path):
                    os.remove(delete_model_path)
            except Exception as e:
                logger.error(f"Fail to delete model {name} at {self.save_dir} caused by exception:{e}.")

        if save_pair:
            try:
                logger.info("--------- updating model --------")
                _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)
            except Exception as e:
                logger.error(f"The following exception:{e} happens when save model to {self.save_dir}.")

    def on_exception(self, exception):
        if self.save_on_exception:
            name = "epoch-{}_step-{}_Exception-{}.pt".format(self.epoch, self.step, exception.__class__.__name__)
            _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)
