# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-18 13:13
from bertsota.parser.biaffine_parser import BlendParser
from bertsota.parser.joint_parser import JointParser

parser = JointParser(cls_parser=BlendParser)
save_dir = 'data/model/blend-joint-sdp'
parser.train(train_file=['data/semeval15/en.dm.train.conllu',
                         'data/semeval15/en.pas.train.conllu',
                         'data/semeval15/en.psd.train.conllu'],
             dev_file=['data/semeval15/en.dm.dev.conllu',
                       'data/semeval15/en.pas.dev.conllu',
                       'data/semeval15/en.psd.dev.conllu'],
             save_dir=save_dir,
             pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt',
             lstm_hiddens=800,
             root='root')
parser.load(save_dir)
parser.evaluate(test_files=['data/semeval15/en.id.dm.conllu',
                            'data/semeval15/en.id.pas.conllu',
                            'data/semeval15/en.id.psd.conllu'], save_dir=save_dir)
parser.evaluate(test_files=['data/semeval15/en.ood.dm.conllu',
                            'data/semeval15/en.ood.pas.conllu',
                            'data/semeval15/en.ood.psd.conllu'], save_dir=save_dir)