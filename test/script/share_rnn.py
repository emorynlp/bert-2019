# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-16 22:03
from bertsota.parser.joint_parser import JointParser

parser = JointParser()
save_dir = 'data/model/1200-joint-sdp'
# parser.train(train_file=['data/semeval15/en.dm.train.conllu',
#                          'data/semeval15/en.pas.train.conllu',
#                          'data/semeval15/en.psd.train.conllu'],
#              dev_file=['data/semeval15/en.dm.dev.conllu',
#                        'data/semeval15/en.pas.dev.conllu',
#                        'data/semeval15/en.psd.dev.conllu'],
#              save_dir=save_dir,
#              lstm_hiddens=1200,
#              pretrained_embeddings_file='data/embedding/fasttext.300d.txt',
#              word_dims=300,
#              root='root')
parser.load(save_dir)
parser.evaluate(test_files=['data/semeval15/en.id.dm.conllu',
                            'data/semeval15/en.id.pas.conllu',
                            'data/semeval15/en.id.psd.conllu'], save_dir=save_dir)
parser.evaluate(test_files=['data/semeval15/en.ood.dm.conllu',
                            'data/semeval15/en.ood.pas.conllu',
                            'data/semeval15/en.ood.psd.conllu'], save_dir=save_dir)