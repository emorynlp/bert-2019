# BERT SOTA Baseline

Source code for paper Establishing Strong Baselines for the New Decade: Sequence Tagging, Syntactic and Semantic Parsing with BERT.

## Requirements

- Python>=3.6
- [mxnet](https://mxnet.apache.org/)>=1.4.1
- [pyhanlp](https://github.com/hankcs/pyhanlp) (optional for preprocessing Chinese)

## Datasets

See [`data`](https://github.com/emorynlp/bert-2019/tree/master/data). All preprocessing scripts are placed in [`test/preprocess`](https://github.com/emorynlp/bert-2019/tree/master/test/preprocess).

## How to Run

All experiment entrypoints are placed in [`test/script`](https://github.com/emorynlp/bert-2019/tree/master/test/script). One example is:

```bash
export PYTHONPATH=.:$PYTHONPATH
python3 test/script/ptb_bert_auto.py
```

