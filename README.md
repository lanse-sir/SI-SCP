# SI-SCP: Learning Structural Information for Syntax-Controlled Paraphrase Generation
## Introdution
This is the pytorch implementation of the paper [Learning Structural Information for Syntax-Controlled Paraphrase Generation](https://aclanthology.org/2022.findings-naacl.160/).
## Requirements
```
python==3.8
pytorch==1.8.0
zss==1.2.0
wandb==0.12.9
```
## Datasets
You can download datasets from [here]. Decompress the data and place the data according to the following structure:
+ SI-SCP
  + autocg
  + SISCP
  + retriever
  + scp_data
    + paranmt
    + quaro
  + str_data
    + paranmt
    + quaro
