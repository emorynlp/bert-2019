# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-13 15:01
import pickle

import numpy as np
from bert_serving.client import BertClient
from pyhanlp import *

CharTable = JClass('com.hankcs.hanlp.dictionary.other.CharTable')

# bc = BertClient(ip='192.168.1.88')  # ip address of the server
bc = BertClient(ip='127.0.0.1')  # ip address of the GPU machine


def embed_last_token(text):
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
                elif t == '[UNK]':
                    t = words[tid][len(buffer)]
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


def embed_sum(text):
    result = bc.encode(text, show_tokens=True)
    # print(result)
    batch = []
    for sent, tensor, tokens in zip(text, result[0], result[1]):
        token_tensor = []
        sent_tensor = []
        tid = 0
        buffer = ''
        words = sent.lower().split()
        for i, t in enumerate(tokens):
            if t == '[CLS]' or t == '[SEP]':
                continue
            else:
                if t.startswith('##'):
                    t = t[2:]
                elif t == '[UNK]':
                    t = words[tid][len(buffer)]
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
            exit()
        batch.append(np.stack(sent_tensor))
    return batch


def generate_bert(path, output, embed_fun=embed_sum):
    print(output)
    total = 0
    with open(path) as src:
        batch = []
        tensor = []
        for line in src:
            line = line.strip()
            if len(line) == 0:
                continue
            batch.append(CharTable.convert(line).replace('—', '-')
                         .replace('‘', '\'')
                         .replace('…', '.')
                         .replace('坜', '壢')
                         .replace('唛', '麦')
                         .replace('ㄅㄆㄇㄈ', '呀呀')
                         .replace('’', '\''))
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


if __name__ == '__main__':
    # generate_bert('data/SemEval-2016/news.test.sent.txt', 'data/SemEval-2016/news.test.bert', embed_fun=embed_sum)
    # generate_bert('data/SemEval-2016/news.valid.sent.txt', 'data/SemEval-2016/news.valid.bert', embed_fun=embed_sum)
    # generate_bert('data/SemEval-2016/news.train.sent.txt', 'data/SemEval-2016/news.train.bert', embed_fun=embed_sum)
    #
    # generate_bert('data/SemEval-2016/text.test.sent.txt', 'data/SemEval-2016/text.test.bert', embed_fun=embed_sum)
    # generate_bert('data/SemEval-2016/text.valid.sent.txt', 'data/SemEval-2016/text.valid.bert', embed_fun=embed_sum)
    # generate_bert('data/SemEval-2016/text.train.sent.txt', 'data/SemEval-2016/text.train.bert', embed_fun=embed_sum)

    generate_bert('data/semeval15/cz.pas.dev.sent.txt', 'data/embedding/bert_base_sum/cz.pas.dev.bert',
                  embed_fun=embed_sum)
    generate_bert('data/semeval15/cz.pas.train.sent.txt', 'data/embedding/bert_base_sum/cz.pas.train.bert',
                  embed_fun=embed_sum)
    generate_bert('data/semeval15/cz.id.pas.sent.txt', 'data/embedding/bert_base_sum/cz.id.pas.bert',
                  embed_fun=embed_sum)

    # generate_bert('data/ctb5.1-pos/dev.short.sent.txt', 'data/embedding/bert_base_sum/ctb.pos.dev.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/ctb5.1-pos/test.short.sent.txt', 'data/embedding/bert_base_sum/ctb.pos.test.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/ctb5.1-pos/train.short.sent.txt', 'data/embedding/bert_base_sum/ctb.pos.train.bert',
    #               embed_fun=embed_sum)

    # generate_bert('data/msra/dev.short.sent.txt', 'data/embedding/bert_base_sum/msra.dev.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/msra/test.short.sent.txt', 'data/embedding/bert_base_sum/msra.test.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/msra/train.short.sent.txt', 'data/embedding/bert_base_sum/msra.train.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/msra/test.auto.short.sent.txt', 'data/embedding/bert_base_sum/msra.test.auto.bert',
    #               embed_fun=embed_sum)

    # generate_bert('data/msra/test.auto.short.sent.txt', 'data/embedding/bert_base_sum/msra.test.auto.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/msra/dev.auto.short.sent.txt', 'data/embedding/bert_base_sum/msra.dev.auto.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/msra/train.auto.short.sent.txt', 'data/embedding/bert_base_sum/msra.train.auto.bert',
    #               embed_fun=embed_sum)

    # generate_bert('data/ctb5/dev.sent.txt', 'data/embedding/bert_base_sum/ctb.dev.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/ctb5/test.sent.txt', 'data/embedding/bert_base_sum/ctb.test.bert',
    #               embed_fun=embed_sum)
    # generate_bert('data/ctb5/train.sent.txt', 'data/embedding/bert_base_sum/ctb.train.bert',
    #               embed_fun=embed_sum)
