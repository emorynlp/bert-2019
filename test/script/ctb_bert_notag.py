# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/ctb-bert-notag'
parser.train(train_file='data/ctb/train.conllx',
             dev_file='data/ctb/dev.conllx', save_dir=save_dir,
             word_dims=300,
             tag_dims=0,
             pretrained_embeddings_file='data/embedding/ctb.fasttext.300.txt',
             bert_path=['data/embedding/bert_base_sum/ctb.train.bert',
                        'data/embedding/bert_base_sum/ctb.dev.bert'],
             root='ROOT')
parser.load(save_dir)
parser.evaluate('data/ctb/test.conllx', save_dir,
                bert_path='data/embedding/bert_base_sum/ctb.test.bert')
