#!/usr/bin/env bash

semeval15=data/semeval15
mkdir -p ${semeval15}
#schemes=( dm pas psd )
#for s in "${schemes[@]}"
#do
#    echo "Converting $1/data/2015/en.${s}.sdp to $semeval15/en.${s}.conllu"
#    python3 -m semstr.convert $1/data/2015/test/en.id.${s}.sdp -f conllu -o ${semeval15} -j en.id.${s}.conllu
#    python3 -m semstr.convert $1/data/2015/test/en.ood.${s}.sdp -f conllu -o ${semeval15} -j en.ood.${s}.conllu
#    python3 -m semstr.convert $1/data/2015/en.${s}.sdp -f conllu -o ${semeval15} -j en.${s}.conllu
#done
#
#echo "Spliting train/dev, using section 21 for devset"
#python3 test/preprocess/split_train_dev.py


echo "Converting $1/data/2015/cz.pas.sdp to $semeval15/cz.pas.conllu"
python3 -m semstr.convert $1/data/2015/test/cz.id.pas.sdp -f conllu -o ${semeval15} -j cz.id.pas.conllu
python3 -m semstr.convert $1/data/2015/cz.pas.sdp -f conllu -o ${semeval15} -j cz.pas.conllu