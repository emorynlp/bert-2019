# coding: utf-8
import pickle

from bertsota.common.data import ParserVocabulary, DataLoader
from bertsota.common.utils import init_logger
from bertsota.parser.dep_parser import SDPParser

if __name__ == '__main__':
    save_dir = 'data/model/bert-dm-noword3'
    parser = SDPParser()
    parser.train(train_file='data/semeval15/en.dm.train.conllu',
                 dev_file='data/semeval15/en.dm.dev.conllu',
                 save_dir=save_dir,
                 bert_path=['data/embedding/bert_base_sum/en.train.bert',
                            'data/embedding/bert_base_sum/en.dev.bert'])
    parser.load(save_dir)
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_file='data/semeval15/en.id.dm.conllu', save_dir=save_dir,
                    bert_path='data/embedding/bert_base_sum/en.id.bert', logger=logger, debug=True)
    parser.evaluate(test_file='data/semeval15/en.ood.dm.conllu', save_dir=save_dir,
                    bert_path='data/embedding/bert_base_sum/en.ood.bert', logger=logger, debug=True)
