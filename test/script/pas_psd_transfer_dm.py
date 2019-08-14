# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-18 23:44
from bertsota.common.utils import init_logger
from bertsota.parser.biaffine_parser import RefineParser, SharedRNNParser
from bertsota.parser.joint_parser import JointParser

for ratio in range(40, 101, 20):
    parser = JointParser(cls_parser=SharedRNNParser)
    save_dir = 'data/model/transfer-pas-psd-dm-{}'.format(ratio)
    print(save_dir)
    parser.train(train_file=['data/semeval15/en.pas.train.{}.conllu'.format(ratio),
                             'data/semeval15/en.psd.train.{}.conllu'.format(ratio),
                             'data/semeval15/en.dm.train.{}.conllu'.format(ratio)],
                 dev_file=['data/semeval15/en.pas.dev.conllu',
                           'data/semeval15/en.psd.dev.conllu',
                           'data/semeval15/en.dm.dev.conllu'],
                 save_dir=save_dir,
                 pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt',
                 lstm_hiddens=800, transfer='data/model/pas-psd-sharernn',
                 root='root')
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_files=['data/semeval15/en.id.pas.conllu',
                                'data/semeval15/en.id.psd.conllu',
                                'data/semeval15/en.id.dm.conllu'], save_dir=save_dir, logger=logger)
    parser.evaluate(test_files=['data/semeval15/en.ood.pas.conllu',
                                'data/semeval15/en.ood.psd.conllu',
                                'data/semeval15/en.ood.dm.conllu'], save_dir=save_dir, logger=logger)
