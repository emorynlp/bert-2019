# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/ptb-baseline'
# parser.train(train_file='data/ptb-dep/train.conllx',
#              dev_file='data/ptb-dep/dev.conllx', save_dir=save_dir,
#              pretrained_embeddings_file='data/embedding/glove/glove.6B.100d.txt',
#              root='root')
parser.load(save_dir)
parser.evaluate('data/ptb-dep/test.conllx', save_dir)
