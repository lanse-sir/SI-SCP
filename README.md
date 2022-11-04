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
You can download datasets from [here](https://drive.google.com/drive/folders/19Iytd_uSzBhDbekSYMzpd2i6O_tjv0dz?usp=sharing). Decompress the data and place the data according to the following structure:
+ SI-SCP
  + autocg
  + SISCP
  + retriever
  + evaluation
  + eval_tools
  + scp_data
    + paranmt
    + quaro
  + str_data
    + paranmt
    + quaro
## Training and Evaluation
The source codes of syntactically controlled paraphrase generation model and syntax template retriever are located in **SISCP** and **retriever** directory, respectively.

# Citation
```
@inproceedings{yang-etal-2022-learning-structural,
    title = "Learning Structural Information for Syntax-Controlled Paraphrase Generation",
    author = "Yang, Erguang  and
      Bai, Chenglin  and
      Xiong, Deyi  and
      Zhang, Yujie  and
      Meng, Yao  and
      Xu, Jinan  and
      Chen, Yufeng",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    year = "2022"}
```
