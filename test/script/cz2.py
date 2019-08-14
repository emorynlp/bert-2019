# coding: utf-8
import pickle

from bertsota.common.data import ParserVocabulary, DataLoader
from bertsota.common.utils import init_logger
from bertsota.parser.dep_parser import SDPParser

if __name__ == '__main__':
    save_dir = 'data/model/cz6'
    parser = SDPParser()
    parser.train(train_file='data/semeval15/cz.pas.train.conllu',
                 dev_file='data/semeval15/cz.pas.dev.conllu',
                 save_dir=save_dir,
                 word_dims=300,
                 pretrained_embeddings_file='data/embedding/cz.fasttext.300.txt')
    parser.load(save_dir)
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_file='data/semeval15/cz.id.pas.conllu', save_dir=save_dir,
                    logger=logger)
