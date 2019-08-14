# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-16 09:19
from collections import Counter

from gluonnlp.embedding import FastText
import gluonnlp as nlp


def load(conllu):
    vocab = set()
    with open(conllu) as src:
        for line in src:
            if line.startswith('#'):
                continue
            info = line.strip().split()
            if info:
                word = info[1].lower()
                vocab.add(word)
    return vocab


def dump(out, vocab):
    embed = FastText(source='wiki-news-300d-1M-subword', load_ngrams=True)
    with open(out, 'w') as out:
        out.write('{} {}\n'.format(len(vocab), 300))
        for word in sorted(vocab):
            out.write('{} {}\n'.format(word, ' '.join(str(x.asscalar()) for x in embed[word])))


if __name__ == '__main__':
    test = load('data/semeval15/en.id.dm.conllu')
    train = load('data/semeval15/en.dm.conllu')
    whole = test | train
    dump('data/embedding/fasttext.300d.txt', whole)
