#!/bin/bash
model=$1
output=$2

python -u predict.py --dev_src ../scp_data/quaro/test/src.txt.bpe --dev_parse ../scp_data/quaro/test/ref_parse.dict --dev_ref ../scp_data/quaro/test/ref.txt --model_file $model --sent $output --pos_vocab ../scp_data/quaro/resource/train.pos_vocab --beam_size 2