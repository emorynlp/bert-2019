# coding: utf-8
from bertsota.common.utils import init_logger
from bertsota.parser.dep_parser import SDPParser

if __name__ == '__main__':
    save_dir = 'data/model/psd'
    parser = SDPParser()
    parser.train(train_file='data/semeval15/en.psd.train.conllu',
                 dev_file='data/semeval15/en.psd.dev.conllu',
                 save_dir=save_dir,
                 pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt')
    parser.load(save_dir)
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_file='data/semeval15/en.id.psd.conllu', save_dir=save_dir, logger=logger, debug=True)
    parser.evaluate(test_file='data/semeval15/en.ood.psd.conllu', save_dir=save_dir, logger=logger, debug=True)
