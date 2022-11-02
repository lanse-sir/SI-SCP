#!/bin/bash
model=$1
output=$2

python -u predict.py --dev_src ../super-quora/bpe-8k/test/src.txt.bpe --dev_parse ../super-quora/bpe-8k/test/tgt_dict.txt --dev_ref ../super-quora/bpe-8k/test/ref.txt --model_file $model --sent $output --pos_vocab ../super-quora/bpe-8k/resource/train.pos_vocab --beam_size 2