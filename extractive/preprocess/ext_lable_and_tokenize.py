# encoding=utf-8

import sys

sys.path.append("../../../../")
import argparse
import time
from others.logging import init_logger
from others.utils import str2bool, mkdir, _get_word_ngrams
from others.logging import logger
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer
import gc
import glob
import hashlib
import json
import os
import re
import subprocess
from os.path import join as pjoin
import torch
from multiprocess import Pool


def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size=3):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    selected = []
    # 把for s in range(len(summary_size)) 改成 for s in range(len(abstract_sent_list)) 消除hard code
    for s in range(len(abstract_sent_list)):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


class BartData(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<sep>", "<cls>"]})
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bos = self.tokenizer.convert_tokens_to_ids('<cls>')
        self.eos = self.tokenizer.convert_tokens_to_ids('<sep>')

    def preprocess(self, src, tgt, sent_labels, real_sent_labels, is_test=False):

        if (not is_test) and len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        _real_sent_labels = [0] * len(src)
        for l in real_sent_labels:
            _real_sent_labels[l] = 1

        # 增加一个real_labels变量 用于保存对应于src_str的label
        real_labels = [_real_sent_labels[i] for i in idxs]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if (not is_test) and len(src) < self.args.min_src_nsents:
            return None

        src_ids_list = [[self.bos] + self.tokenizer.encode(' '.join(sent), add_special_tokens=False) + [self.eos]
                        for sent in src]
        src_ids = [0]
        for src_id in src_ids_list:
            if len(src_ids) + len(src_id) < self.args.max_src_ntokens:
                src_ids += src_id
            else:
                break
        src_ids += [2]
        assert len(src_ids) <= 1024
        cls_ids = [i for i, t in enumerate(src_ids) if t == self.bos]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_txt = [' '.join(tt) for tt in tgt]
        src_txt = [original_src_txt[i] for i in idxs]

        return src_ids, cls_ids, src_txt[:len(cls_ids)], tgt_txt, sent_labels[:len(cls_ids)], real_labels


import nltk


# lower
def sent_to_words(sent_list):
    return [sent.split() for sent in sent_list]


def format_to_bart(args):
    inst, is_test = args
    article_id = inst["article_id"]
    # for CNNDM
    # text = nltk.sent_tokenize(inst['text'])
    # summary = inst['summary'].split("\n")
    # for other dataset
    text = inst["text"]
    summary = inst["summary"]
    source, tgt = sent_to_words(text), sent_to_words(summary)
    sent_labels = greedy_selection(source[:parsed_args.max_src_nsents], tgt)
    real_sent_labels = greedy_selection(source, tgt)
    b_data = bart.preprocess(source, tgt, sent_labels, real_sent_labels, is_test=is_test)
    if b_data is None:
        return None
    src_ids, cls_ids, src_txt, tgt_txt, sent_labels, real_labels = b_data
    b_data_dict = {"article_id": article_id, "text_id": src_ids, "cls_ids": cls_ids,
                   "summary": tgt_txt, 'text': src_txt, "labels": sent_labels}
    return b_data_dict


def format_to_bart_mp():
    datasets = ['val', 'test', "train"]
    for corpus_type in datasets:
        a_lst = []
        for jsonl_f in glob.glob(pjoin(parsed_args.raw_path, f'{corpus_type}.jsonl')):
            real_name = jsonl_f.split('/')[-1]
            a_lst.append((jsonl_f, pjoin(parsed_args.save_path, real_name.replace('jsonl', 'id.jsonl'))))
        if len(a_lst) != 1:
            raise RuntimeError(f"文件夹里面包含多个命中为 *{corpus_type}*.jsonl 的文件或无此文件")

        jsonl_file, save_file = a_lst[0]
        logger.info('Processing %s' % jsonl_file)
        jsonl_insts = load_jsonl(jsonl_file)
        is_test = [(corpus_type == 'test')] * len(jsonl_insts)

        with Pool(parsed_args.n_cpu) as p:
            formatted_insts = p.map(format_to_bart, zip(jsonl_insts, is_test))

        logger.info('Saving to %s' % save_file)

        origin_len = len(formatted_insts)
        formatted_insts = [inst for inst in formatted_insts if inst is not None]
        print(f"pop {origin_len - len(formatted_insts)} insts")
        with open(os.path.join(save_file), "w") as wf:
            for inst in formatted_insts:
                wf.write(json.dumps(inst) + "\n")
        gc.collect()


if __name__ == '__main__':
    base_dir = "../datasets"
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default=None)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--has_label', type=str2bool, default=False)
    parser.add_argument('--min_src_nsents', default=1, type=int)
    parser.add_argument('--max_src_nsents', default=50, type=int)
    parser.add_argument('--min_src_ntokens_per_sent', default=3, type=int)
    parser.add_argument('--max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('--min_tgt_ntokens', default=5, type=int)
    parser.add_argument('--max_tgt_ntokens', default=500, type=int)
    parser.add_argument('--max_src_ntokens', default=1024, type=int)
    parser.add_argument('--dataset', default="CNNDM")

    parser.add_argument("--lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--log_file', default='../logs/default.log')

    parser.add_argument('--n_cpu', default=36, type=int)

    parsed_args = parser.parse_args()

    if parsed_args.raw_path is None:
        parsed_args.raw_path = base_dir
    assert os.path.exists(parsed_args.raw_path)
    if parsed_args.save_path is None:
        parsed_args.save_path = os.path.join(base_dir, parsed_args.dataset)
    if not os.path.exists(parsed_args.save_path):
        mkdir(parsed_args.save_path)

    logger = init_logger(parsed_args.log_file)
    bart = BartData(parsed_args)

    logger.info(time.clock())
    format_to_bart_mp()
    logger.info(time.clock())
