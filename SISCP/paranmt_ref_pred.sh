#!/bin/bash
model=$1
output=$2

python -u predict.py --dev_src ../paranmt-50w/bpe-15k/test/test_src.bpe --dev_parse ../paranmt-50w/bpe-15k/test/test_tgt_parse_dict.bpe --dev_ref ../paranmt-50w/test/test_tgt.txt --pos_vocab ../paranmt-50w/bpe-15k/resource/train.pos_vocab --model_file $model --sent $output