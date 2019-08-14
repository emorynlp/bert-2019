# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import DepParser

parser = DepParser()
save_dir = 'data/model/ctb-old'
parser.train(train_file='data/ctb5.1-dep/train.conllx',
             dev_file='data/ctb5.1-dep/dev.conllx', save_dir=save_dir,
             word_dims=300,
             # interpolation=0.2,
             pretrained_embeddings_file='data/embedding/ctb.fasttext.300.txt',
             root='ROOT')
parser.load(save_dir)
parser.evaluate('data/ctb5.1-dep/test.conllx', save_dir)
