## Datasets

See the appendix section in our paper. The expected folder structure is as follows.

```
data
├── SemEval-2016
│   ├── news.test.conllu
│   ├── news.train.conllu
│   ├── news.valid.conllu
│   ├── text.test.conllu
│   ├── text.train.conllu
│   └── text.valid.conllu
├── conll03
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── ctb5.1-dep
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── ctb5.1-pos
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── ptb-dep
│   ├── dev.conllx
│   ├── test.conllx
│   └── train.conllx
├── semeval15
│   ├── en.dm.dev.conllu
│   ├── en.dm.train.conllu
│   ├── en.id.dm.conllu
│   ├── en.id.pas.conllu
│   ├── en.id.psd.conllu
│   ├── en.ood.dm.conllu
│   ├── en.ood.pas.conllu
│   ├── en.ood.psd.conllu
│   ├── en.pas.dev.conllu
│   ├── en.pas.train.conllu
│   ├── en.psd.dev.conllu
│   └── en.psd.train.conllu
└── wsj-pos
    ├── dev.tsv
    ├── test.tsv
    └── train.tsv
```

- For sdp data, the `.conllu` files are converted with the semstr. For example,

  ```bash
  $ pip3 install semstr
  $ python3 -m semstr.convert sdp2014_2015/data/2015/en.psd.sdp -f conllu -o . -j en.psd.conllu
  ```

- Alternatively, `test/preprocess/convert_sdp.sh` and `test/preprocess/convert_sdp_cn.sh` batch the conversion.

- Some sentences in ctb and ptb are too long for BERT (<=512 tokens), which need to be splitted. See `test/preprocess/limit_length.py` for detail.

- For Chinese characters normalization, [pyhanlp](https://github.com/hankcs/pyhanlp) is required, which can be installed through `pip3 install pyhanlp`.

- The BERT embeddings can be abtained through [bert-as-service](https://github.com/hanxiao/bert-as-service) or gluonnlp. See `test/preprocess/en_bert.py` , `test/preprocess/cn_bert.py` and `test/preprocess/enbert.py` for detail.