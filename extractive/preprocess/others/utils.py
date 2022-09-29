import argparse
import os
from os.path import exists


def get_data_path(mode, label_type):
    paths = {}
    if mode == 'train':
        paths['train'] = 'data/' + label_type + '/bert.train.jsonl'
        paths['val'] = 'data/' + label_type + '/bert.val.jsonl'
    else:
        paths['test'] = 'data/' + label_type + '/bert.test.jsonl'
    return paths


def get_rouge_path(label_type):
    if label_type == 'others':
        data_path = 'data/' + label_type + '/bert.test.jsonl'
    else:
        data_path = 'data/' + label_type + '/test.jsonl'
    dec_path = 'dec'
    ref_path = 'ref'
    mkdir(ref_path)
    mkdir(dec_path)
    return data_path, dec_path, ref_path


def mkdir(dir_name):
    if not exists(dir_name):
        os.makedirs(dir_name)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x