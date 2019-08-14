# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-13 15:01
import os
import pickle
import numpy as np

from bert_serving.client import BertClient

bc = BertClient(ip='127.0.0.1')  # ip address of the GPU machine
# bc = BertClient(ip='192.168.1.88')


def embed(text):
    result = bc.encode(text, show_tokens=True)
    # print(result)
    batch = []
    for sent, tensor, tokens in zip(text, result[0], result[1]):
        valid = []
        tid = 0
        buffer = ''
        words = sent.lower().split()
        for i, t in enumerate(tokens):
            if t == '[CLS]' or t == '[SEP]':
                continue
            else:
                if t.startswith('##'):
                    t = t[2:]
                buffer += t
                if buffer == words[tid]:
                    valid.append(i)
                    buffer = ''
                    tid += 1
        # print(len(valid))
        # exit()
        if len(valid) != len(sent.split()) or tid != len(words):
            print(valid)
            print(sent.split())
            print(result[1])
        batch.append(tensor[valid, :])
    return batch


def embed_sum(text, cased=False):
    result = bc.encode(text, show_tokens=True)
    # print(result)
    batch = []
    for sent, tensor, tokens in zip(text, result[0], result[1]):
        token_tensor = []
        sent_tensor = []
        tid = 0
        buffer = ''
        if not cased:
            sent = sent.lower()
        words = sent.split()
        for i, t in enumerate(tokens):
            if t == '[CLS]' or t == '[SEP]':
                continue
            else:
                if t.startswith('##'):
                    t = t[2:]
                buffer += t
                token_tensor.append(tensor[i, :])
                if buffer == words[tid]:
                    sent_tensor.append(np.stack(token_tensor).mean(axis=0))
                    token_tensor = []
                    buffer = ''
                    tid += 1
        # print(len(valid))
        # exit()
        if tid != len(words) or len(sent_tensor) != len(words):
            print(sent.split())
            print(tokens)
            exit(1)
        batch.append(np.stack(sent_tensor))
    return batch


def make_bert_for(path, output, embed_fun=embed_sum):
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
                tensor.extend(embed_fun(batch))
                total += len(batch)
                print(total)
                batch = []
        if len(batch):
            tensor.extend(embed_fun(batch))
            total += len(batch)
            print(total)
        with open(output, 'wb') as f:
            pickle.dump(tensor, f)


def make_bert_text(path, output, embed_fun=embed_sum, cased=False):
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
                tensor.extend(embed_fun(batch, cased))
                total += len(batch)
                print(total)
                batch = []
        if len(batch):
            tensor.extend(embed_fun(batch, cased))
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

    # make_bert_text('data/conll03/dev.sent.txt', 'data/embedding/bert_large_sum/conll03.dev.bert', cased=True)
    # make_bert_text('data/conll03/test.sent.txt', 'data/embedding/bert_large_sum/conll03.test.bert', cased=True)
    # make_bert_text('data/conll03/train.sent.txt', 'data/embedding/bert_large_sum/conll03.train.bert', cased=True)

    make_bert_text('data/conll03/dev.sent.txt', 'data/embedding/bert_base_sum/conll03.dev.bert', cased=False)
    make_bert_text('data/conll03/test.sent.txt', 'data/embedding/bert_base_sum/conll03.test.bert', cased=False)
    make_bert_text('data/conll03/train.sent.txt', 'data/embedding/bert_base_sum/conll03.train.bert', cased=False)
