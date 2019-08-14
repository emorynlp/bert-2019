# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-13 15:01
import os
import pickle

import numpy as np
from bert_embedding import BertEmbedding

from bertsota.common.utils import mxnet_prefer_gpu

bert = BertEmbedding(model='bert_24_1024_16',
                     dataset_name='book_corpus_wiki_en_cased',
                     max_seq_length=270,
                     ctx=mxnet_prefer_gpu())


def embed_bert(sents):
    result = bert.embedding(sents)
    return [np.stack(s[2]) for s in result]


def make_bert_for(path, output):
    print(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    total = 0
    with open(path) as src:
        batch = []
        tensor = []
        for line in src:
            line = line.strip()
            if line.startswith('# text = '):
                text = line[len('# text = '):]
                batch.append(text)
            if len(batch) and len(batch) % 100 == 0:
                tensor.extend(embed_bert(batch))
                total += len(batch)
                print(total)
                batch = []
        if len(batch):
            tensor.extend(embed_bert(batch))
            total += len(batch)
            print(total)
        with open(output, 'wb') as f:
            pickle.dump(tensor, f)


def make_bert_text(path, output):
    print(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    total = 0
    with open(path) as src:
        batch = []
        tensor = []
        for line in src:
            line = line.strip()
            if len(line) == 0:
                continue
            batch.append(line)
            if len(batch) and len(batch) % 100 == 0:
                tensor.extend(embed_bert(batch))
                total += len(batch)
                print(total)
                batch = []
        if len(batch):
            tensor.extend(embed_bert(batch))
            total += len(batch)
            print(total)
        with open(output, 'wb') as f:
            pickle.dump(tensor, f)


if __name__ == '__main__':
    # make_bert_for('data/semeval15/en.psd.dev.conllu', 'data/embedding/bert_large/en.dev.bert', embed_fun=embed)
    # make_bert_for('data/semeval15/en.psd.train.conllu', 'data/embedding/bert_large/en.train.bert', embed_fun=embed)
    # make_bert_for('data/semeval15/en.ood.psd.conllu', 'data/embedding/bert_large/en.ood.bert', embed_fun=embed)
    # make_bert_for('data/semeval15/en.id.psd.conllu', 'data/embedding/bert_large/en.id.bert', embed_fun=embed)
    #
    # make_bert_for('data/semeval15/en.psd.dev.conllu', 'data/embedding/bert_large_sum/en.dev.bert')
    # make_bert_for('data/semeval15/en.psd.train.conllu', 'data/embedding/bert_large_sum/en.train.bert')
    # make_bert_for('data/semeval15/en.ood.psd.conllu', 'data/embedding/bert_large_sum/en.ood.bert')
    # make_bert_for('data/semeval15/en.id.psd.conllu', 'data/embedding/bert_large_sum/en.id.bert')

    # make_bert_text('data/wsj-pos/dev.sent.txt', 'data/embedding/bert_base_sum/wsj.dev.bert')
    # make_bert_text('data/wsj-pos/test.sent.txt', 'data/embedding/bert_base_sum/wsj.test.bert')
    # make_bert_text('data/wsj-pos/train.sent.txt', 'data/embedding/bert_base_sum/wsj.train.bert')

    # make_bert_text('data/wsj-pos/dev.sent.txt', 'data/embedding/bert_large_sum/wsj.dev.bert')
    # make_bert_text('data/wsj-pos/test.sent.txt', 'data/embedding/bert_large_sum/wsj.test.bert')
    # make_bert_text('data/wsj-pos/train.sent.txt', 'data/embedding/bert_large_sum/wsj.train.bert')

    # make_bert_text('data/conll03/dev.sent.txt', 'data/embedding/bert_large_sum/conll03.dev.bert', cased=True)
    # make_bert_text('data/conll03/test.sent.txt', 'data/embedding/bert_large_sum/conll03.test.bert', cased=True)
    # make_bert_text('data/conll03/train.sent.txt', 'data/embedding/bert_large_sum/conll03.train.bert', cased=True)

    # make_bert_text('data/conll03/dev.sent.txt', 'data/embedding/bert_large_sum/conll03.dev.bert')
    # make_bert_text('data/conll03/test.sent.txt', 'data/embedding/bert_large_sum/conll03.test.bert')
    # make_bert_text('data/conll03/train.sent.txt', 'data/embedding/bert_large_sum/conll03.train.bert')

    # make_bert_text('data/ontonotes-en/dev.sent.txt', 'data/embedding/bert_large_sum/ontonotes-en.dev.bert')
    # make_bert_text('data/ontonotes-en/test.sent.txt', 'data/embedding/bert_large_sum/ontonotes-en.test.bert')
    # make_bert_text('data/ontonotes-en/train.sent.txt', 'data/embedding/bert_large_sum/ontonotes-en.train.bert')

    make_bert_text('data/wsj-pos/dev.sent.txt', 'data/embedding/bert_large_cased/wsj.dev.bert')
    make_bert_text('data/wsj-pos/test.sent.txt', 'data/embedding/bert_large_cased/wsj.test.bert')
    make_bert_text('data/wsj-pos/train.short.sent.txt', 'data/embedding/bert_large_cased/wsj.train.short.bert')

    # make_bert_text('data/wsj-pos/dev.sent.txt', 'data/embedding/bert_base_cased/wsj.dev.bert')
    # make_bert_text('data/wsj-pos/test.sent.txt', 'data/embedding/bert_base_cased/wsj.test.bert')
    # make_bert_text('data/wsj-pos/train.sent.txt', 'data/embedding/bert_base_cased/wsj.train.bert')

    # make_bert_text('data/wsj-pos/dev.sent.txt', 'data/embedding/bert_large_sum/wsj.dev.bert')
    # make_bert_text('data/wsj-pos/test.sent.txt', 'data/embedding/bert_large_sum/wsj.test.bert')
    # make_bert_text('data/wsj-pos/train.sent.txt', 'data/embedding/bert_large_sum/wsj.train.bert')
