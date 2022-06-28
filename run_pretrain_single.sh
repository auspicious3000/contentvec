#!/bin/bash

expdir=./tmp
mkdir -p $expdir


HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py  \
    --config-dir ./contentvec/config/contentvec \
    --config-name contentvec \
    hydra.run.dir=${expdir} \
    task.data=./metadata \
    task.label_dir=./label \
    task.labels=["km"] \
    task.spk2info=./spk2info.dict \
    task.crop=true \
    dataset.train_subset=train \
    dataset.valid_subset=valid \
    dataset.num_workers=10 \
    dataset.max_tokens=500000 \
    checkpoint.keep_best_checkpoints=10 \
    criterion.loss_weights=[10,1e-5] \
    model.label_rate=50 \
    model.encoder_layers_1=3 \
    model.logit_temp_ctr=0.1 \
    model.ctr_layers=[-6] \
    model.extractor_mode="default" \
    optimization.update_freq=[1] \
    optimization.max_update=100000 \
    lr_scheduler.warmup_updates=8000 