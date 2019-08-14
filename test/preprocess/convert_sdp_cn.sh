#!/usr/bin/env bash

semeval16=data/SemEval-2016
rm -f ${semeval16}/*.conllu
python3 -m semstr.convert data/SemEval-2016/train/news.train.conll -f conllu -o ${semeval16} -j news.train.conllu
python3 -m semstr.convert data/SemEval-2016/train/text.train.conll -f conllu -o ${semeval16} -j text.train.conllu

python3 -m semstr.convert data/SemEval-2016/test/news.test.conll -f conllu -o ${semeval16} -j news.test.conllu
python3 -m semstr.convert data/SemEval-2016/test/text.test.conll -f conllu -o ${semeval16} -j text.test.conllu

python3 -m semstr.convert data/SemEval-2016/validation/news.valid.conll -f conllu -o ${semeval16} -j news.valid.conllu
python3 -m semstr.convert data/SemEval-2016/validation/text.valid.conll -f conllu -o ${semeval16} -j text.valid.conllu