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
            word = line.split()[1]
            vocab.add(word)
    return vocab


test = extract_words('data/semeval15/cz.id.pas.conllu')
train = extract_words('data/semeval15/cz.pas.conllu')

with open('data/semeval15/cz.word.txt', 'w') as out:
    for word in sorted(test | train):
        out.write(word)
        out.write('\n')
