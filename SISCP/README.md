# Syntactically Controlled Paraphrase Model

## Training

Paranmt-small dataset:

```
python -u train.py --run_name paranmt --model_config bpe_15k_paranmt-50w.yaml
```

QQP-Pos dataset:
```
python -u train.py --run_name qqp --model_config bpe_8k_qqp.yaml
```


## Inference

Paranmt-small dataset:

```
bash paranmt_ref_pred.sh <model_file> <output_file>
```

QQP-Pos dataset:
```
bash qqp_ref_pred.sh <model_file> <output_file>
```

## Diverse Paraphrase Generation
Paranmt-small dataset:

```
python -u generate_with_temp.py --dev_parse <retrieved_template_file> --dev_src ../scp_data/paranmt/test/test_src.bpe --model_file <model_file> --sent <output_file>
```

QQP-Pos dataset:
```
python -u generate_with_temp.py --dev_parse <retrieved_template_file> --dev_src ../scp_data/quaro/test/src.txt.bpe --model_file <model_file> --sent <output_file>
```

