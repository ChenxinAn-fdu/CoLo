import sys
import argparse
import os
import json
import torch
from time import time
from datetime import timedelta
from os.path import join, exists

from torch import optim
from torch.optim import Adam
from transformers import AutoTokenizer

from model.dataloader import CoLoExtLoader
from model.model import CoLoExtModel, BaselineExtModel
from model.metrics import CoLoLoss, SummDevMetric, ExtBCELoss, GetSysOut
from model.callback import LrCallback, SaveModelCallback
from fastNLP import Trainer, Tester

def get_data_path(mode, data_path):
    paths = {}
    if mode == "train":
        paths['train'] = f'{data_path}/train.id.jsonl'
        paths['val'] = f'{data_path}/val.id.jsonl'
    paths['test'] = f'{data_path}/test.id.jsonl'
    return paths


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def configure_training(args):
    devices = [int(gpu) for gpu in args.gpus.split(',')]
    params = {}
    params['label_type'] = args.label_type
    params['batch_size'] = args.batch_size
    params['accum_count'] = args.accum_count
    params['max_lr'] = args.max_lr
    params['warmup'] = args.warmup
    params['validate_every'] = args.validate_every
    params['n_epochs'] = args.n_epochs
    return devices, params


def test_model_baseline(args):
    # load dataset
    data_paths = get_data_path(args.mode, f"datasets/{args.dataset}")
    datasets = CoLoExtLoader(args.pad_id, args.ext_num).process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    test_set = datasets.datasets['test']
    article_dict = {}
    with open(data_paths['test']) as f:
        for line in f:
            article = json.loads(line)
            article_dict[article['article_id']] = {'text': article['text'], 'summary': article['summary']}
    # only need 1 gpu for testing
    device = int(args.gpus)

    model = BaselineExtModel(args, article_dict=article_dict)
    model_ckpt = torch.load(args.warmup_ckpt)
    model.load_state_dict(model_ckpt)
    model.article_dict = article_dict
    # configure testing
    test_metric = GetSysOut(save_path="checkpoints/baseline.out")
    tester = Tester(data=test_set, model=model, metrics=[test_metric],
                    batch_size=args.batch_size, device=device)
    tester.test()


def test_model_CoLo(args):
    models = os.listdir(args.save_path)
    # load dataset
    data_paths = get_data_path(args.mode, f"datasets/{args.dataset}")
    datasets = CoLoExtLoader(args.pad_id, args.ext_num).process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    test_set = datasets.datasets['test']
    article_dict = {}
    with open(data_paths['test']) as f:
        for line in f:
            article = json.loads(line)
            article_dict[article['article_id']] = {'text': article['text'], 'summary': article['summary']}
    # only need 1 gpu for testing
    device = int(args.gpus)

    for cur_model in models:
        print('Current model is {}'.format(cur_model))
        # load model
        model = CoLoExtModel(args, article_dict=article_dict)
        model_ckpt = torch.load(join(args.save_path, cur_model))
        model.load_state_dict(model_ckpt)
        model.article_dict = article_dict
        model.block_trigram = args.block_trigram
        model.ext_num = args.ext_num
        # configure testing
        test_metric = GetSysOut(save_path=os.path.join(args.save_path, cur_model))
        tester = Tester(data=test_set, model=model, metrics=[test_metric],
                        batch_size=args.batch_size, device=device)
        tester.test()


def train_model_CoLo(args):
    # check if the data_path and save_path exists
    data_paths = get_data_path(args.mode, f"datasets/{args.dataset}")
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    # load summarization datasets
    datasets = CoLoExtLoader(args.pad_id, args.ext_num).process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    train_set = datasets.datasets['train']
    dev_set = datasets.datasets['val']
    # configure training
    devices, train_params = configure_training(args)
    article_dict = {}
    with open(data_paths['train']) as f:
        for line in f:
            article = json.loads(line)
            article_dict[article['article_id']] = {'text': article['text'], 'summary': article['summary']}
    with open(data_paths['val']) as f:
        for line in f:
            article = json.loads(line)
            article_dict[article['article_id']] = {'text': article['text'], 'summary': article['summary']}
    # configure model
    pretrain_model = torch.load(args.warmup_ckpt, map_location="cpu")
    model = CoLoExtModel(args, article_dict=article_dict)
    model.load_state_dict(pretrain_model)
    model.article_dict = article_dict
    model.ext_num = args.ext_num
    model.metric = args.metric
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0)
    callbacks = [LrCallback(args), SaveModelCallback(args.save_path, top=3, warmup_ckpt=None, only_param=True)]
    criterion = CoLoLoss()
    dev_metric = SummDevMetric()
    trainer = Trainer(train_data=train_set, model=model, optimizer=optimizer, dev_data=dev_set,
                      loss=criterion, batch_size=args.batch_size, metrics=dev_metric,
                      update_every=args.accum_count, n_epochs=args.n_epochs, validate_every=args.validate_every,
                      print_every=50, save_path=args.save_path, device=devices, callbacks=callbacks)

    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()


def train_model_baseline(args):
    # check if the data_path and save_path exists
    data_paths = get_data_path(args.mode, f"datasets/{args.dataset}")
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    # load summarization datasets
    datasets = CoLoExtLoader(args.pad_id, args.ext_num).process(data_paths)
    print('Information of dataset is:')
    print(datasets)
    train_set = datasets.datasets['train']
    dev_set = datasets.datasets['val']
    # configure training
    devices, train_params = configure_training(args)
    # configure model
    article_dict = {}
    with open(data_paths['train']) as f:
        for line in f:
            article = json.loads(line)
            article_dict[article['article_id']] = {'text': article['text'], 'summary': article['summary']}
    with open(data_paths['val']) as f:
        for line in f:
            article = json.loads(line)
            article_dict[article['article_id']] = {'text': article['text'], 'summary': article['summary']}

    model = BaselineExtModel(args, article_dict)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0)
    callbacks = [LrCallback(args),
                 SaveModelCallback(default_warmup_dir, top=1, warmup_ckpt=default_warmup_ckpt, only_param=True)]
    criterion = ExtBCELoss()
    dev_metrics = SummDevMetric()
    trainer = Trainer(train_data=train_set, dev_data=dev_set, model=model, optimizer=optimizer,
                      loss=criterion, batch_size=args.batch_size, metrics=dev_metrics,
                      update_every=args.accum_count, n_epochs=args.n_epochs, validate_every=args.validate_every,
                      print_every=50, device=devices, callbacks=callbacks)

    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()
    print("========== If  the warmed-up ends with exception, please kill the next process =========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of bart ext version'
    )
    parser.add_argument('--mode', required=True,
                        help='training or testing of BertSum', type=str)

    parser.add_argument('--label_type', default='greedy',
                        help='greedy/limit', type=str)

    # example for gpus input: '0,1,2,3'
    parser.add_argument('--gpus', required=True,
                        help='available gpus for training(separated by commas)', type=str)

    parser.add_argument('--batch_size', default=36,
                        help='the training batch size', type=int)
    parser.add_argument('--accum_count', default=1,
                        help='number of updates steps to accumulate before performing a backward/update pass.',
                        type=int)
    parser.add_argument('--max_lr', default=2e-5,
                        help='max learning rate for warm up', type=float)
    parser.add_argument('--n_epochs', default=5,
                        help='total number of training epochs', type=int)
    parser.add_argument('--validate_every', default=8000,
                        help='number of update steps for checkpoint', type=int)
    parser.add_argument('--save_path', default="checkpoints", help='root of the model', type=str)
    parser.add_argument('--warmup', default=True, type=str2bool)
    parser.add_argument('--warmup_ckpt', default=None)
    parser.add_argument('--version', default="large-cnn", choices=["large", "large-cnn"])

    parser.add_argument('--dataset', default='CNNDM')
    parser.add_argument('--block_trigram', default=True, type=str2bool)
    parser.add_argument('--ext_num', default=5, type=int)
    parser.add_argument('--metric', default="rouge")
    parser.add_argument('--lr_warmup_steps', default=10000,
                        help='warm up steps for training', type=int)

    # pad value for BART (change this if using other ptm)
    parser.add_argument('--pad_id', default=1, type=int)
    args = parser.parse_args()

    default_warmup_dir = f"warmed_up/{args.dataset}/"
    default_warmup_ckpt = "pretrain.ext.pt"
    if args.warmup_ckpt is None:
        args.warmup_ckpt = f'{default_warmup_dir}/{default_warmup_ckpt}'

    if args.mode == 'train':
        if args.warmup:
            print('Training process of ext model with only BCELoss')
            train_model_baseline(args)
        else:
            print('Training process of CoLo ext ')
            train_model_CoLo(args)

    else:
        if not os.path.exists(args.warmup_ckpt):
            raise FileNotFoundError(f"we can not load a warmed up checkpoint {args.warmup_ckpt}")
        print('Testing process of CoLo ext ')
        test_model_CoLo(args)


