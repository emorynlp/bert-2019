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


dev = extract_words('data/SemEval-2016/text.valid.conllu')
test = extract_words('data/SemEval-2016/text.test.conllu')
train = extract_words('data/SemEval-2016/text.train.conllu')

with open('data/SemEval-2016/text.word.txt', 'w') as out:
    for word in sorted(dev | test | train):
        out.write(word)
        out.write('\n')
