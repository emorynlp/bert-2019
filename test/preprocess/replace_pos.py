# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-02-14 01:10

def replace(old, new, out):
    print(out)
    with open(old) as old, open(new) as new, open(out, 'w') as out:
        for o, n in zip(old, new):
            o = o.strip()
            n = n.strip()
            if len(o) == 0:
                out.write('\n')
                continue
            oc = o.split()
            nc = n.split()
            oc[1] = oc[1].replace('-LRB-', '(')
            oc[1] = oc[1].replace('-RRB-', ')')
            oc[1] = oc[1].replace('-LCB-', '{')
            oc[1] = oc[1].replace('-RCB-', '}')
            nc[1] = nc[1].replace('\/', '/')
            nc[1] = nc[1].replace('\/', '/')
            nc[1] = nc[1].replace('\*', '*')
            if oc[1] != nc[1] and not (oc[1].isnumeric() and nc[1] == '#'):
                print(o)
                print(n)
                exit(1)
            oc[2] = nc[2]
            oc[3] = nc[3]
            oc[4] = nc[4]
            out.write('\t'.join(oc))
            out.write('\n')


replace('data/ptb-dep/dev.conllx', '/Users/nlp/Downloads/wsj-dep/dev.conllx', 'data/ptb-dep/dev.auto.conllx')
replace('data/ptb-dep/test.conllx', '/Users/nlp/Downloads/wsj-dep/test.conllx', 'data/ptb-dep/test.auto.conllx')
replace('data/ptb-dep/train.conllx', '/Users/nlp/Downloads/wsj-dep/train.conllx', 'data/ptb-dep/train.auto.conllx')
