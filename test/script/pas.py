# coding: utf-8
from bertsota.common.utils import init_logger
from bertsota.parser.dep_parser import SDPParser

if __name__ == '__main__':
    save_dir = 'data/model/pas'
    parser = SDPParser()
    # parser.train(train_file='data/semeval15/en.pas.train.conllu',
    #              dev_file='data/semeval15/en.pas.dev.conllu',
    #              save_dir=save_dir,
    #              pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt')
    parser.load(save_dir)
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_file='data/semeval15/en.id.pas.conllu', save_dir=save_dir, logger=logger, debug=True)
    parser.evaluate(test_file='data/semeval15/en.ood.pas.conllu', save_dir=save_dir, logger=logger, debug=True)
