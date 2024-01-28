# PIP: Parse-Instructed Prefix for Syntactically Controlled Paraphrase Generation
Official source code repository for the ACL 2023 Findings paper: **"PIP: Parse-Instructed Prefix for Syntactically Controlled Paraphrase Generation"** by *Yixin Wan and Kuan-Hao Huang and Kai-Wei Chang*.

Link to full paper: https://arxiv.org/abs/2305.16701.

* To build a conda environment for running experiments, cd into the current repository and run the command:
```
conda create --name <env> --file requirements.txt
```
* To train the baseline model with seq2seq training, run:
```
sh ./scripts/train_seq2seq.sh
```
* To train the baseline model with prefix tuning, run:
```
sh ./scripts/train_prefix_tuning.sh
```
* To train the PIP-direct model, run:
```
sh ./scripts/train_pip_direct.sh
```
* To train the PIP-indirect model, run:
```
sh ./scripts/train_pip_indirect.sh
```
