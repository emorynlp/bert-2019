# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-28 19:17

def eval(train, pred):
    vocab = set()
    with open(train) as train:
        for line in train:
            line: str = line.strip()
            if len(line) == 0:
                continue
            vocab.add(line.split()[0])

    total = 0
    oov_right = 0
    with  open(pred) as pred:
        for line in pred:
            line: str = line.strip()
            if len(line) == 0:
                continue
            cells = line.split()
            if cells[0] not in vocab:
                total += 1
                if cells[1] == cells[2]:
                    oov_right += 1
    print('%.2f\n' % (oov_right / total * 100))


if __name__ == '__main__':
    eval('data/wsj-pos/train.tsv', 'data/model/wsj-pos-ge-fe4/test.tsv')
    # eval('data/ctb5.1-pos/train.tsv', 'data/model/ctb-pos-noword4/test.tsv')
