# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-12 15:25
import mxnet as mx

from bertsota.common.utils import mxnet_prefer_gpu
from bertsota.tagger.corpus import NLPTaskDataFetcher
from bertsota.tagger.embeddings import WordEmbeddings, CharLMEmbeddings, StackedEmbeddings
from bertsota.tagger.sequence_tagger_model import SequenceTagger
from bertsota.tagger.sequence_tagger_trainer import SequenceTaggerTrainer

columns = {0: 'text', 1: 'pos'}
corpus = NLPTaskDataFetcher.fetch_column_corpus('data/wsj-pos',
                                                columns,
                                                train_file='train.tsv',
                                                test_file='test.tsv',
                                                dev_file='dev.tsv')
# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
with mx.Context(mxnet_prefer_gpu()):
    embedding_types = [
        WordEmbeddings('data/embedding/glove/glove.6B.100d.txt'),
        # CharLMEmbeddings('data/model/lm-news-forward'),
        # CharLMEmbeddings('data/model/lm-news-backward'),
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
    trainer.train('data/model/wsj-pos', learning_rate=0.1, mini_batch_size=32, max_epochs=150,
                  embeddings_in_memory=True)
