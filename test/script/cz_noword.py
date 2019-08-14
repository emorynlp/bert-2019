# coding: utf-8
import pickle

from bertsota.common.data import ParserVocabulary, DataLoader
from bertsota.common.utils import init_logger
from bertsota.parser.dep_parser import SDPParser

if __name__ == '__main__':
    save_dir = 'data/model/cz-noword2'
    parser = SDPParser()
    parser.train(train_file='data/semeval15/cz.pas.train.conllu',
                 dev_file='data/semeval15/cz.pas.dev.conllu',
                 save_dir=save_dir,
                 bert_path=['data/embedding/bert_base_sum/cz.pas.train.bert',
                            'data/embedding/bert_base_sum/cz.pas.dev.bert'])
    parser.load(save_dir)
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_file='data/semeval15/cz.id.pas.conllu', save_dir=save_dir,
                    bert_path='data/embedding/bert_base_sum/cz.id.pas.bert',
                    logger=logger)
