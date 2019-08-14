# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-18 23:44
from bertsota.common.utils import init_logger
from bertsota.parser.biaffine_parser import RefineParser, CNNParser
from bertsota.parser.joint_parser import JointParser

parser = JointParser(cls_parser=CNNParser)
save_dir = 'data/model/maxpool-cnn-refine-joint-sdp'
# parser.train(train_file=['data/semeval15/en.dm.train.conllu',
#                          'data/semeval15/en.pas.train.conllu',
#                          'data/semeval15/en.psd.train.conllu'],
#              dev_file=['data/semeval15/en.dm.dev.conllu',
#                        'data/semeval15/en.pas.dev.conllu',
#                        'data/semeval15/en.psd.dev.conllu'],
#              save_dir=save_dir,
#              pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt',
#              lstm_hiddens=800,
#              root='root')
parser.load(save_dir)
logger = init_logger(save_dir, 'test.log')
parser.evaluate(test_files=['data/semeval15/en.id.dm.conllu',
                            'data/semeval15/en.id.pas.conllu',
                            'data/semeval15/en.id.psd.conllu'], save_dir=save_dir, logger=logger)
parser.evaluate(test_files=['data/semeval15/en.ood.dm.conllu',
                            'data/semeval15/en.ood.pas.conllu',
                            'data/semeval15/en.ood.psd.conllu'], save_dir=save_dir, logger=logger)
