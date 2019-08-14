# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-07 20:45
from bertsota.parser.dep_parser import SDPParser

parser = SDPParser()
save_dir = 'data/model/news-baseline5'
parser.train(train_file='data/SemEval-2016/news.train.conllu',
             dev_file='data/SemEval-2016/news.valid.conllu', save_dir=save_dir,
             word_dims=300,
             pretrained_embeddings_file='data/embedding/news.fasttext.300.txt',
             root='root')
parser.load(save_dir)
parser.evaluate('data/SemEval-2016/news.test.conllu', save_dir, chinese=True)
