# BERT SOTA Baseline

Source code for paper [Establishing Strong Baselines for the New Decade: Sequence Tagging, Syntactic and Semantic Parsing with BERT](https://arxiv.org/pdf/1908.04943.pdf), to be published in The Thirty-Third International Flairs Conference proceedings.

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

### References

If you use this repository in your research, please kindly cite our FLAIRS-33 paper:

```bibtex
@inproceedings{bertbaseline,
  title={Establishing Strong Baselines for the New Decade: Sequence Tagging, Syntactic and Semantic Parsing with BERT},
  author={He, Han and Choi, Jinho},
  booktitle={The Thirty-Third International Flairs Conference},
  year={2020}
}
```

