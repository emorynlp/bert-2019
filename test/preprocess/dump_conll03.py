# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-17 00:20
import os

from bertsota.tagger.corpus import iob_iobes, iob2


def dump(src, out_word, out_tag):
    with open(src) as src, open(out_word, 'w') as out_word, open(out_tag, 'w') as out_tag:
        words = []
        tags = []
        for line in src:
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                out_word.write(' '.join(words))
                out_word.write('\n')
                iob2(tags)
                out_tag.write(' '.join(iob_iobes(tags)))
                out_tag.write('\n')
                words = []
                tags = []
                continue
            word, _, _, tag = line.split()
            words.append(word)
            tags.append(tag)

        if len(words):
            out_word.write(' '.join(words))
            out_word.write('\n')
            iob2(tags)
            out_tag.write(' '.join(iob_iobes(tags)))
            out_tag.write('\n')


if __name__ == '__main__':
    root = '/Users/nlp/PythonProjects/tf_ner/data/conll03'
    for src, out in zip(['train', 'dev', 'test'], ['train', 'testa', 'testb']):
        dump(os.path.join('data/conll03', src + '.tsv'),
             os.path.join(root, out + '.words.txt'),
             os.path.join(root, out + '.tags.txt'),)
