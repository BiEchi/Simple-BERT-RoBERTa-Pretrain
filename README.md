# 🔥 Simple RoBERTa Pre-training from Scratch

HuggingFace reproduction of the BERT/RoBERTa pretraining from scratch, with memory optimization using DeepSpeed and BF16. Our implementation takes 4 days on 2*8 NVIDIA ADA-arch GPUs. You just need a script to run the whole pretraining.

## Environment Setup

First, clone this repository and install the dependencies.

```
git clone git@github.com:BiEchi/RoBERTa-Pretrain.git
conda create --name roberta python==3.9.0
conda activate roberta
pip install torch
pip install transformers
pip install wandb
```

Next, make sure the pre-training (`run_mlm.py`) and fine-tuning (`run_glue.py`) scripts are up-to-date with huggingface. You can refresh them by

```
cd RoBERTa-Pretrain
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/text-classification/run_glue.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/language-modeling/run_mlm.py
```

## Choose Model

On the architecture side, RoBERTa is exactly the same as BERT, except for its larger vocabulary size. We define the configs of various model sizes of both BERT and RoBERTa in the [configs](https://github.com/BiEchi/RoBERTa-Pretrain/tree/main/configs) directory. When launching the script, you simply need to change these two lines in [pretrain.sh](https://github.com/BiEchi/RoBERTa-Pretrain/blob/main/pretrain.sh):

```
export MODEL=configs/roberta_medium.json # change this config to what you want
export TOKENIZER=roberta-base # or bert-base-uncased for the BERT family
```

According to Google's [BERT releases](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8), a medium sized model should have a config of Layer=8, Hidden=512, #AttnHeads=8, and IntermediateSize=2048. We follow this config for all sizes of both models.

## Datasets

We use the same datasets as BERT (English Wikipedia and Book Corpus) to pre-train. I released the reproduction of this dataset at [JackBAI/bert_pretrain_datasets](https://huggingface.co/datasets/JackBAI/bert_pretrain_datasets). Note that this dataset is not pre-tokenizer or grouped so it might take some time the first time you load this dataset.

You can also use any other datasets you want.

## Hyper-parameters

We utilized DeepSpeed ZeRO-2 and BF16 for performance optimization.

Other training configuration: 

| Parameter            | Value     |
|----------------------|-----------|
| WARMUP_STEPS         | 1800      |
| LR_DECAY             | linear    |
| ADAM_EPS             | 1e-6      |
| ADAM_BETA1           | 0.9       |
| ADAM_BETA2           | 0.98      |
| ADAM_WEIGHT_DECAY    | 0.01      |
| PEAK_LR              | 1e-3      |

## Evaluation Results

We pre-trained a RoBERTa-Medium model for 30k steps with a batch size of 8,192, and released the checkpoint at [JackBAI/roberta-medium](https://huggingface.co/JackBAI/roberta-medium). We achieve very similar performance as the official BERT-Medium release on GLUE:

| Model           | MRPC-F1 | STS-B-Pearson | SST-2-Acc | QQP-F1 | MNLI-m | MNLI-mm | QNLI-Acc | WNLI-Acc | RTE-Acc |
|----------------|---------|---------------|-----------|--------|--------|---------|----------|----------|---------|
| RoBERTa-medium (ours)         |  83.6    | 82.7          | 89.7      | 89.0   | 79.7   | 80.1    | 89.3     | 31.0     | 57.4    |
| [BERT-medium](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8) | 86.3 | 87.7 | 88.9 | 89.4 | 80.6 | 81.0 | 89.2 | 29.6 | 63.9 |

Evaluation Scores Curve (AVG of scores) during pretraining:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62927c2e56fedc76e396b3ca/p6KK_6-oXCyhAjdZh1WQ_.png)

For both stats above we don't report CoLA scores as it's pretty unstable. The raw CoLA scores are:

| Step     | 1500 | 3000 | 6000 | 9000 | 13500 | 18000 | 24000 | 30000 |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
CoLA | 1.7 | 13.5 | 29.2 | 31.4 | 31.1 | 24.1 | 29.0 | 20.0 |


