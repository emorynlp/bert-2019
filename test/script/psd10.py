# coding: utf-8
from bertsota.common.data import ParserVocabulary, DataLoader
from bertsota.common.utils import init_logger
from bertsota.parser.dep_parser import SDPParser

for p in range(20, 100, 20):
    save_dir = 'data/model/bert-base-psd{}'.format(p)
    parser = SDPParser()
    parser.train(train_file='data/semeval15/en.psd.train.{}.conllu'.format(p),
                 dev_file='data/semeval15/en.psd.dev.conllu',
                 save_dir=save_dir,
                 pretrained_embeddings_file='data/embedding/glove.6B.100d.shrinked.txt',
                 bert_path=['data/embedding/bert_base_sum/en.train.bert',
                            'data/embedding/bert_base_sum/en.dev.bert'])
    parser.load(save_dir)
    logger = init_logger(save_dir, 'test.log')
    parser.evaluate(test_file='data/semeval15/en.id.psd.conllu', bert_path='data/embedding/bert_base_sum/en.id.bert',
                    save_dir=save_dir, logger=logger)
    parser.evaluate(test_file='data/semeval15/en.ood.psd.conllu', bert_path='data/embedding/bert_base_sum/en.ood.bert',
                    save_dir=save_dir, logger=logger)
