# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import SDPParser

parser = SDPParser()
save_dir = 'data/model/text-noword3'
parser.train(train_file='data/SemEval-2016/text.train.conllu',
             dev_file='data/SemEval-2016/text.valid.conllu', save_dir=save_dir,
             bert_path=['data/embedding/bert_base_sum/text.train.bert',
                        'data/embedding/bert_base_sum/text.valid.bert'],
             root='root')
parser.load(save_dir)
parser.evaluate('data/SemEval-2016/text.test.conllu', save_dir, bert_path='data/embedding/bert_base_sum/text.test.bert',
                chinese=True)
