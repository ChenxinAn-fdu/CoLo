import os

import json
import torch
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase
from rouge import rouge_score
from fastNLP import logger


class LossMetric(MetricBase):
    def __init__(self, pred=None, labels=None, mask=None):
        super(LossMetric, self).__init__()
        self._init_param_map(pred=pred, labels=labels, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')
        self.avg_loss = 0.0
        self.nsamples = 0

    def evaluate(self, pred, labels, mask):
        batch_size = pred.size(0)
        loss = self.loss_func(pred, labels.float())
        loss = (loss * mask.float()).sum()
        self.avg_loss += loss
        self.nsamples += batch_size

    def get_metric(self, reset=True):
        self.avg_loss = self.avg_loss / self.nsamples
        eval_result = {'loss': self.avg_loss}
        if reset:
            self.avg_loss = 0
            self.nsamples = 0
        return eval_result


class ExtBCELoss(LossBase):

    def __init__(self, pred=None, labels=None, mask=None):
        super(ExtBCELoss, self).__init__()
        self._init_param_map(pred=pred, labels=labels, mask=mask)
        self.loss_func = torch.nn.BCELoss(reduction='none')

    def get_loss(self, pred, labels, mask):
        loss = self.loss_func(pred, labels.float())
        loss = (loss * mask.float()).sum() / pred.size(0)
        return loss


class CoLoLoss(LossBase):
    def __init__(self, margin=0.01, pred=None, labels=None, mask=None, score=None):
        super(CoLoLoss, self).__init__()
        self._init_param_map(pred=pred, labels=labels, mask=mask, score=score)
        self.loss_bce = torch.nn.BCELoss(reduction='none')
        self.margin = margin
        self.loss_func = torch.nn.MarginRankingLoss(margin)

    def ranking_loss(self, score):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)
        return TotalLoss

    def get_loss(self, pred, labels, mask, score):
        bce_loss = self.loss_bce(pred, labels.float())
        bce_loss = (bce_loss * mask.float()).sum() / mask.sum()
        return self.ranking_loss(score) + bce_loss


class SummDevMetric(MetricBase):
    """
    this metric is used for dev set, we do not call `pyrouge` for efficiency
    """

    def __init__(self):
        super(SummDevMetric, self).__init__()
        self.rouge_list = []

    def evaluate(self, rouge):
        self.rouge_list.append(rouge.mean())

    def get_metric(self, reset=True):
        rouge_avg = sum(self.rouge_list) / len(self.rouge_list)
        self.rouge_list = []
        return {"rouge": rouge_avg.item()}


class GetSysOut(MetricBase):
    """
    this metric is used to generate system output
    """

    def __init__(self, save_path):
        super(GetSysOut, self).__init__()
        self.save_path = save_path
        self.sys_outs = []

    def evaluate(self, output, gold_ref):
        bsz = len(output)
        for i in range(bsz):
            self.sys_outs.append(
                {'sys_out': ' '.join(output[i]).replace("\n", ''), 'ref_out': ' '.join(gold_ref[i]).replace("\n", '')})

    def get_metric(self, reset=True):
        result_path = self.save_path.replace("checkpoints", "results").replace(".pt", ".jsonl")
        result_dir = os.path.dirname(result_path)
        os.makedirs(result_dir, exist_ok=True)
        logger.info(f"======= save system output to {result_path} ======= ")
        with open(result_path, "w") as f:
            for i, inst in enumerate(self.sys_outs):
                if i == 0:
                    logger.info(f"the first line: {json.dumps(inst)}")
                print(json.dumps(inst), file=f)
        logger.info(f"======= write file ends there total {len(self.sys_outs)} samples ======= ")
        logger.info(" the `evaluation/eval_with_pyrouge.py` script will  produce final results ")
        return {"rouge": None}
