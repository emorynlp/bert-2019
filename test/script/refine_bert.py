# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-16 22:03
from bertsota.common.utils import init_logger
from bertsota.parser.biaffine_parser import RefineParser
from bertsota.parser.joint_parser import JointParser

parser = JointParser(cls_parser=RefineParser)
save_dir = 'data/model/refine-bert'
parser.train(train_file=['data/semeval15/en.dm.train.conllu',
                         'data/semeval15/en.pas.train.conllu',
                         'data/semeval15/en.psd.train.conllu'],
             dev_file=['data/semeval15/en.dm.dev.conllu',
                       'data/semeval15/en.pas.dev.conllu',
                       'data/semeval15/en.psd.dev.conllu'],
             save_dir=save_dir,
             lstm_hiddens=1200,
             pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt',
             bert_path=['data/embedding/en.train.bert', 'data/embedding/en.dev.bert'],
             root='root')
parser.load(save_dir)
logger = init_logger(save_dir, 'test.log')
parser.evaluate(test_files=['data/semeval15/en.id.dm.conllu',
                            'data/semeval15/en.id.pas.conllu',
                            'data/semeval15/en.id.psd.conllu'],
                bert_path='data/embedding/en.id.bert',
                save_dir=save_dir, logger=logger)
parser.evaluate(test_files=['data/semeval15/en.ood.dm.conllu',
                            'data/semeval15/en.ood.pas.conllu',
                            'data/semeval15/en.ood.psd.conllu'],
                bert_path='data/embedding/en.ood.bert',
                save_dir=save_dir, logger=logger)
