# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-08 14:16

def extract_words(conll):
    vocab = set()
    with open(conll) as src:
        for line in src:
            line = line.strip()
            if len(line) == 0:
                continue
            word = line.split()[0]
            vocab.add(word)
    return vocab


dev = extract_words('data/ctb5.1-pos/dev.tsv')
test = extract_words('data/ctb5.1-pos/test.tsv')
train = extract_words('data/ctb5.1-pos/train.tsv')

with open('data/ctb5.1-pos/word.txt', 'w') as out:
    for word in sorted(dev | test | train):
        out.write(word)
        out.write('\n')
