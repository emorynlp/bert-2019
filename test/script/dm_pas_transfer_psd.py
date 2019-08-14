# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-18 23:44
from bertsota.common.utils import init_logger
from bertsota.parser.biaffine_parser import RefineParser, SharedRNNParser
from bertsota.parser.joint_parser import JointParser

for ratio in range(20, 101, 20):
    save_dir = 'data/model/transfer-dm-pas-psd-{}'.format(ratio)
    print(save_dir)
    parser = JointParser(cls_parser=SharedRNNParser)
    parser.train(train_file=['data/semeval15/en.dm.train.{}.conllu'.format(ratio),
                             'data/semeval15/en.pas.train.{}.conllu'.format(ratio),
                             'data/semeval15/en.psd.train.{}.conllu'.format(ratio)],
                 dev_file=['data/semeval15/en.dm.dev.conllu',
                           'data/semeval15/en.pas.dev.conllu',
                           'data/semeval15/en.psd.dev.conllu'],
                 save_dir=save_dir,
                 pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt',
                 lstm_hiddens=800, transfer='data/model/dm-pas-sharernn',
                 root='root')
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_files=['data/semeval15/en.id.dm.conllu',
                                'data/semeval15/en.id.pas.conllu',
                                'data/semeval15/en.id.psd.conllu'], save_dir=save_dir, logger=logger)
    parser.evaluate(test_files=['data/semeval15/en.ood.dm.conllu',
                                'data/semeval15/en.ood.pas.conllu',
                                'data/semeval15/en.ood.psd.conllu'], save_dir=save_dir, logger=logger)
