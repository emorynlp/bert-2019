# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-08 14:16
import os


def extract_sent(conll, word_idx=1, char=True):
    out = os.path.splitext(conll)[0] + '.sent.txt'
    print(out)
    max_len = 0
    with open(conll) as src, open(out, 'w') as out:
        sent = []
        for line in src:
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                out.write(' '.join(sent))
                sent_len = len(''.join(sent)) if char else len(sent)
                if sent_len >= max_len:
                    # print(' '.join(sent))
                    max_len = len(''.join(sent)) if char else len(sent)
                sent = []
                out.write('\n')
                continue
            word = line.split()[word_idx]
            if not char:
                word = word.replace('ö', '*')
                word = word.replace('Ì', '*')
                word = word.replace('￥', '$')  # OOV char
            sent.append(word)
        if len(sent):
            sent_len = len(''.join(sent)) if char else len(sent)
            out.write(' '.join(sent))
            if sent_len >= max_len:
                # print(' '.join(sent))
                max_len = len(''.join(sent)) if char else len(sent)
            sent = []
            out.write('\n')
    print(max_len)


if __name__ == '__main__':
    # extract_sent('data/SemEval-2016/news.test.conllu')
    # extract_sent('data/SemEval-2016/news.valid.conllu')
    # extract_sent('data/SemEval-2016/news.train.conllu')
    #
    # extract_sent('data/SemEval-2016/text.test.conllu')
    # extract_sent('data/SemEval-2016/text.valid.conllu')
    # extract_sent('data/SemEval-2016/text.train.conllu')

    extract_sent('data/semeval15/cz.id.pas.conllu')
    extract_sent('data/semeval15/cz.pas.dev.conllu')
    extract_sent('data/semeval15/cz.pas.train.conllu')

    # extract_sent('data/wsj-pos/dev.tsv', word_idx=0)
    # extract_sent('data/wsj-pos/test.tsv', word_idx=0)
    # extract_sent('data/wsj-pos/train.tsv', word_idx=0)

    # extract_sent('data/conll03/dev.tsv', word_idx=0)
    # extract_sent('data/conll03/test.tsv', word_idx=0)
    # extract_sent('data/conll03/train.tsv', word_idx=0)

    # extract_sent('data/ctb5.1-pos/dev.short.tsv', word_idx=0)
    # extract_sent('data/ctb5.1-pos/test.short.tsv', word_idx=0)
    # extract_sent('data/ctb5.1-pos/train.short.tsv', word_idx=0)

    # extract_sent('data/msra/dev.short.tsv', word_idx=0, cn=False)
    # extract_sent('data/msra/test.short.tsv', word_idx=0, cn=False)
    # extract_sent('data/msra/train.short.tsv', word_idx=0, cn=False)

    # extract_sent('data/msra-ner/dev.tsv', word_idx=0, cn=False)
    # extract_sent('data/msra-ner/test.tsv', word_idx=0, cn=False)
    # extract_sent('data/msra-ner/train.tsv', word_idx=0, cn=False)

    # extract_sent('data/ontonotes-en/dev.tsv', word_idx=1, cn=False)
    # extract_sent('data/ontonotes-en/test.tsv', word_idx=1, cn=False)
    # extract_sent('data/ontonotes-en/train.tsv', word_idx=1, cn=False)

    # extract_sent('data/wsj-pos/train.short.tsv', word_idx=0, cn=True)

    # extract_sent('data/msra/test.auto.short.tsv', word_idx=0, char=True)
    # extract_sent('data/msra/dev.auto.short.tsv', word_idx=0, char=True)
    # extract_sent('data/msra/train.auto.short.tsv', word_idx=0, char=True)

    # extract_sent('data/ctb5/dev.conll')
    # extract_sent('data/ctb5/test.conll')
    # extract_sent('data/ctb5/train.conll')
