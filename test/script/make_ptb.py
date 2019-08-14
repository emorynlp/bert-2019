# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-23 18:50
import re


def text(sent: str):
    return re.sub('\W+', ' ',
                  sent.split('\n')[2][len('# text = '):].strip()
                  .replace('-LRB-', '').replace('-RRB-', '')
                  .replace('-LCB-', '').replace('-RCB-', ''))


def align(ptb, sdp, out):
    with open(ptb) as ptb, open(sdp) as sdp, open(out, 'w') as out:
        sdp = sdp.read().split('\n\n')
        ptb = dict((text(sent), sent) for sent in ptb.read().split('\n\n') if '# text = ' in sent)
        for sent in sdp:
            if '# text = ' not in sent:
                print(sent)
                continue
            out.writelines(sent[:3])
            t = text(sent)
            ps = ptb[t]
            for p, s in zip(ps.split('\n'), sent.split('\n')):
                pc = p.split('\t')
                sc = s.split('\t')
                out.write('\t'.join(sc[:6]))
                out.write('\t')
                out.write('\t'.join(pc[6:]))
                out.write('\n')
            out.write('\n')


if __name__ == '__main__':
    align('data/ptb-sdp/ptb.train.conllu', 'data/semeval15/en.dm.train.conllu', 'data/ptb-sdp/ptb.sdp.train.conllu')
    align('data/ptb-sdp/ptb.dev.conllu', 'data/semeval15/en.dm.dev.conllu', 'data/ptb-sdp/ptb.sdp.dev.conllu')
    align('data/ptb-sdp/ptb.test.conllu', 'data/semeval15/en.id.dm.conllu', 'data/ptb-sdp/ptb.sdp.test.conllu')
