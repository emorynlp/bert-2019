# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-06 20:10
from bertsota.common.data import split_train_dev

# for data in 'dm', 'pas', 'psd':
#     split_train_dev('data/semeval15/en.{}.conllu'.format(data), 'data/semeval15/en.{}'.format(data))

with open('data/semeval15/cz.pas.conllu') as src, open('data/semeval15/cz.pas.train.conllu', 'w') as out:
    sents = src.read().split('\n\n')
    out.write("\n\n".join(s for s in sents[:int(len(sents) * 0.9)]))

with open('data/semeval15/cz.pas.conllu') as src, open('data/semeval15/cz.pas.dev.conllu', 'w') as out:
    sents = src.read().split('\n\n')
    out.write("\n\n".join(s for s in sents[int(len(sents) * 0.9):]))
