# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-16 09:19


def load(conllu, idx=1, lower=True):
    vocab = set()
    with open(conllu) as src:
        for line in src:
            if line.startswith('#'):
                continue
            info = line.strip().split()
            if info:
                word = info[idx]
                if lower:
                    word = word.lower()
                vocab.add(word)
    return vocab


def dump(pret, out, vocab):
    with open(pret) as src, open(out, 'w') as out:
        iv = set()
        for line in src:
            cells = line.strip().split()
            if len(cells) > 2:
                word, data = cells[0], cells[1:]
                word = word.lower()
                if word in vocab:
                    out.write(line)
                    iv.add(word)
            else:
                out.write(line)
        print('IV Rate {} / {}'.format(len(iv), len(vocab)))


if __name__ == '__main__':
    # test = load('data/semeval15/en.id.dm.conllu')
    # train = load('data/semeval15/en.dm.conllu')
    # whole = test | train
    # dump('data/embedding/glove/glove.6B.100d.txt', 'data/embedding/glove.6B.100d.shrinked.txt', whole)
    dev = load('data/msra-ner/dev.tsv', idx=0, lower=False)
    test = load('data/msra-ner/test.tsv', idx=0, lower=False)
    train = load('data/msra-ner/train.tsv', idx=0, lower=False)
    whole = test | train | dev
    dump('data/embedding/character.vec', 'data/embedding/msra-character.vec', whole)
