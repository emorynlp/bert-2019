# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/ctb-noword5'
parser.train(train_file='data/ctb5/train.conll',
             dev_file='data/ctb5/dev.conll', save_dir=save_dir,
             bert_path=['data/embedding/bert_base_sum/ctb.train.bert',
                        'data/embedding/bert_base_sum/ctb.dev.bert'],
             root='ROOT')
parser.load(save_dir)
parser.evaluate('data/ctb5/test.conll', save_dir,
                bert_path='data/embedding/bert_base_sum/ctb.test.bert')
