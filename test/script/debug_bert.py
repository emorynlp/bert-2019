# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 23:13
import pickle

from bertsota.common.data import ParserVocabulary, DepDataLoader

with open('data/embedding/bert_base_sum/ptb.dep.dev.bert', 'rb') as f:
    bert = pickle.load(f)
    print(len(bert))

vocab = ParserVocabulary('data/ptb-dep/train.conllx',
                         'data/embedding/glove/glove.6B.100d.debug.txt',
                         min_occur_count=2)

data_loader = DepDataLoader(
    'data/ptb-dep/train.conllx', 50, vocab, bert='data/embedding/bert_base_sum/ptb.dep.dev.bert')
next(data_loader.get_batches(100, False))

with open('data/embedding/bert_base_sum/en.train.bert', 'rb') as src:
    tensor = pickle.load(src)
    print(len(tensor))
