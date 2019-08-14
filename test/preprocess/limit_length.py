# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-12 23:32

def limit_ctb(src, out, max_len):
    with open(src) as src, open(out, 'w') as out:
        cur_len = 0
        for line in src:
            if len(line.strip()) == 0:
                out.write('\n')
                continue
            out.write(line)
            word, pos = line.strip().split()
            if cur_len + len(word) >= max_len and (pos == 'PU' or word in '！，,。？、、：；／）/)' or pos.startswith('E')):
                out.write('\n')
                cur_len = 0
            else:
                cur_len += len(word)


def limit_msr(src, out, max_len):
    with open(src) as src, open(out, 'w') as out:
        cur_len = 0
        for line in src:
            if len(line.strip()) == 0:
                out.write('\n')
                continue
            out.write(line)
            word, pos = line.strip().split()
            if cur_len + 1 >= max_len and (word in '！，,。？、、：；／）/)'):
                out.write('\n')
                cur_len = 0
            else:
                cur_len += 1


def limit_onto(src, out, max_len):
    with open(src) as src, open(out, 'w') as out:
        cur_len = 0
        for line in src:
            if len(line.strip()) == 0:
                out.write('\n')
                continue
            out.write(line)
            cells = line.strip().split()
            word = cells[1]
            pos = cells[7]
            if cur_len + len(word) >= max_len and (pos == 'p'):
                out.write('\n')
                cur_len = 0
            else:
                cur_len += len(word)


def limit_wsj(src, out, max_len):
    with open(src) as src, open(out, 'w') as out:
        cur_len = 0
        for line in src:
            if len(line.strip()) == 0:
                out.write('\n')
                continue
            out.write(line)
            cells = line.strip().split()
            word = cells[1]
            if cur_len + len(word) >= max_len and (word == ','):
                out.write('\n')
                cur_len = 0
            else:
                cur_len += len(word)


# limit('data/ctb5.1-pos/dev.tsv', 'data/ctb5.1-pos/dev.short.tsv', max_len=128)
# limit('data/ctb5.1-pos/test.tsv', 'data/ctb5.1-pos/test.short.tsv', max_len=128)
# limit('data/ctb5.1-pos/train.tsv', 'data/ctb5.1-pos/train.short.tsv', max_len=128)

# limit_ctb('data/msra/dev.tsv', 'data/msra/dev.short.tsv', max_len=128)
# limit_ctb('data/msra/test.tsv', 'data/msra/test.short.tsv', max_len=128)
# limit_ctb('data/msra/train.tsv', 'data/msra/train.short.tsv', max_len=128)

limit_msr('data/msra/dev.auto.tsv', 'data/msra/dev.auto.short.tsv', max_len=128)
limit_msr('data/msra/test.auto.tsv', 'data/msra/test.auto.short.tsv', max_len=128)
limit_msr('data/msra/train.auto.tsv', 'data/msra/train.auto.short.tsv', max_len=128)

# limit_onto('data/ontonotes-en/dev.tsv', 'data/ontonotes-en/dev.short.tsv', max_len=128)
# limit_onto('data/ontonotes-en/test.tsv', 'data/ontonotes-en/test.short.tsv', max_len=128)
# limit_onto('data/ontonotes-en/train.tsv', 'data/ontonotes-en/train.short.tsv', max_len=128)
# limit_wsj('data/wsj-pos/train.tsv', 'data/wsj-pos/train.short.tsv', max_len=256)
