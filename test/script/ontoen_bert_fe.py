# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-12 17:10
import os

import mxnet as mx

from bertsota.common.utils import mxnet_prefer_gpu
from bertsota.tagger.corpus import NLPTaskDataFetcher, NLPTask

# get training, test and dev data
from bertsota.tagger.embeddings import WordEmbeddings, StackedEmbeddings, BERTEmbeddings, CharLMEmbeddings
from bertsota.tagger.sequence_tagger_model import SequenceTagger
from bertsota.tagger.sequence_tagger_trainer import SequenceTaggerTrainer

model_path = 'data/model/ontoen-bert-fe'
columns = {0: 'id', 1: 'text', 2: 'lemma', 3: 'ner'}
corpus = NLPTaskDataFetcher.fetch_column_corpus('data/ontonotes-en',
                                                columns,
                                                train_file='train.tsv',
                                                test_file='test.tsv',
                                                dev_file='dev.tsv',
                                                tag_to_biloes='ner',
                                                source_scheme='ioblu')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
with mx.Context(mxnet_prefer_gpu()):
    embedding_types = [

        # WordEmbeddings('data/embedding/glove/glove.6B.100d.txt'),
        BERTEmbeddings(['data/embedding/bert_large_sum/ontonotes-en.train.bert',
                        'data/embedding/bert_large_sum/ontonotes-en.dev.bert',
                        'data/embedding/bert_large_sum/ontonotes-en.test.bert']),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use contextual string embeddings
        CharLMEmbeddings('data/model/lm-news-forward'),
        CharLMEmbeddings('data/model/lm-news-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_crf=True)

    # 6. initialize trainer
    trainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

    # 7. start training
    trainer.train(model_path,
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150,
                  embeddings_in_memory=True)

    tagger.load_parameters(os.path.join(model_path, 'model.bin'), ctx=mxnet_prefer_gpu())
    trainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)
    test_score, test_fp, test_result = trainer.evaluate(corpus.test, model_path,
                                                        evaluation_method='span-F1',
                                                        embeddings_in_memory=True)
    print(test_score)
