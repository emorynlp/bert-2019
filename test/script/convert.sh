#!/usr/bin/env bash

python3 /Users/nlp/PythonProjects/semstr/semstr/convert.py /Users/nlp/Documents/语料库/sdp2014_2015/data/2015/test/en.ood.pas.sdp -f conllu -j en.ood.pas -o /Users/nlp/PythonProjects/biaffinesdp/data/semeval15
python3 /Users/nlp/PythonProjects/semstr/semstr/convert.py /Users/nlp/Documents/语料库/sdp2014_2015/data/2015/test/en.ood.dm.sdp -f conllu -j en.ood.dm -o /Users/nlp/PythonProjects/biaffinesdp/data/semeval15
#python3 /Users/nlp/PythonProjects/semstr/semstr/convert.py /Users/nlp/Documents/语料库/sdp2014_2015/data/2015/en.dm.sdp -f conllu -j en.dm -o /Users/nlp/PythonProjects/biaffinesdp/data/semeval15
#python3 /Users/nlp/PythonProjects/semstr/semstr/convert.py /Users/nlp/Documents/语料库/sdp2014_2015/data/2015/en.pas.sdp -f conllu -j en.pas -o /Users/nlp/PythonProjects/biaffinesdp/data/semeval15