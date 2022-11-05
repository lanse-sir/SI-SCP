# Syntactic Template Retriever

## Training
Paranmt-small dataset:
```
bash paranmt_train.sh
```

QQP-Pos dataset:
```
bash quaro_train.sh
```


## Inference

Paranmt-small dataset:
```
python -u inference.py --model_file <model_file> --input_sent ../str_data/paranmt/test_src.txt --input_pos ../str_data/paranmt/test_src_parse-4.txt --corpus ../str_data/paranmt/parse-4_set-15.txt --tgt_parse ../str_data/paranmt/test_tgt_parse-4.txt
```

QQP-Pos dataset:
```
python -u inference.py --model_file <model_file> --input_sent ../str_data/quaro/test_src.txt --input_pos ../str_data/quaro/test_src_parse-4.txt --corpus ../str_data/quaro/parse-4_set-15.txt --tgt_parse ../str_data/quaro/test_tgt_parse-4.txt
```
