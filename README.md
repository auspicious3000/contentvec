# ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers 

This repository provides the official PyTorch implementation of [ContentVec](https://arxiv.org/abs/2204.09224).

This is a short video that explains the main concepts of our work. If you find this work useful and use it in your research, please consider citing our paper.

[![ContentVec](./assets/cover.png)](https://youtu.be/wow2DRuJ69c/)


## Pre-trained models 
|Model | Classes |  |
|---|---|---|
|ContentVec_legacy | 100 | [download](https://ibm.box.com/s/0moa6xqexvphmkpnaabk9sg3fwxfh4ly)
|ContentVec | 100 | [download](https://ibm.box.com/s/vy3mmba6kdhbg0jdvq1uqluynpmcfpsy)
|ContentVec_legacy | 500 | [download](https://ibm.box.com/s/r9ex6xjoaeesd8xfttm4kvccvnn529uw)
|ContentVec | 500 | [download](https://ibm.box.com/s/mbj6p8ruv14xbkbcoqlwdus7rpx9p6pp)


## Load a model without setting up code repo
```
ckpt_path = "/path/to/the/checkpoint_best_legacy.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```


## Train a new model
### Data preparation
Download the [zip file](https://ibm.box.com/s/3snda5zxz9o0tjw8vc072hykqxtub3nm) consisting of the following files:
- `{train,valid}.tsv` waveform list files in metadata
- `{train,valid}.km` frame-aligned pseudo label files in labels
- `dict.km.txt` a dummy dictionary in labels
- `spk2info.dict` a dictionary mapping from speaker id to speaker embedding in metadata

Modify the root directory in the `{train,valid}.tsv` waveform list files

### Setup code repo
Follow steps in `setup.sh` to setup the code repo

### Pretrain ContentVec
Use `run_pretrain_single.sh` to run on a single node

Use `run_pretrain_multi.sh` and the corresponding slurm template to run on multiple GPUs and nodes
