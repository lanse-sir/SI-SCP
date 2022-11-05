#!/bin/bash
model=$1
output=$2

python -u predict.py --dev_src ../scp_data/paranmt/test/test_src.bpe --dev_parse ../scp_data/paranmt/test/test_tgt_parse_dict.bpe --dev_ref ../scp_data/paranmt/test/test_tgt.txt --pos_vocab ../scp_data/paranmt/resource/train.pos_vocab --model_file $model --sent $output --beam_size 2