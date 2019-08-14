# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/ptb-noword-auto2'
parser.train(train_file='data/ptb-dep/train.auto.conllx',
             dev_file='data/ptb-dep/dev.auto.conllx', save_dir=save_dir,
             bert_path=['data/embedding/bert_base_sum/ptb.dep.train.bert',
                        'data/embedding/bert_base_sum/ptb.dep.dev.bert'],
             root='root')
parser.load(save_dir)
parser.evaluate('data/ptb-dep/test.auto.conllx', save_dir, bert_path='data/embedding/bert_base_sum/ptb.dep.test.bert')
