#!/bin/bash

stage=2
stop_stage=2

PYTHON_ENVIRONMENT=contentvec
CONDA_ROOT=/mnt/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh

cwd=$(pwd)
FAIRSEQ=${cwd}/fairseq/fairseq
CODE=${cwd}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Install conda environment..."
    
    conda create --name ${PYTHON_ENVIRONMENT} python=3.7 -y
fi

conda activate ${PYTHON_ENVIRONMENT}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Install fairseq and other dependencies..."
    
    if [ ! -d fairseq ]; then
        git clone https://github.com/pytorch/fairseq.git --branch main --single-branch
    fi
    cd ${cwd}/fairseq
    # checkout the fairseq version to use
    git reset --hard 0b21875e45f332bedbcc0617dcf9379d3c03855f

    if [ $(pip freeze | grep fairseq | wc -l ) -gt 0 ]; then
        echo "Already installed fairseq. Skip..."
    else
        echo "Install fairseq..."
        python -m pip install --editable ./
    fi
    # optionally do this if related error occurs
    python setup.py build_ext --inplace
    
    python -m pip install scipy
    python -m pip install soundfile
    python -m pip install praat-parselmouth
    python -m pip install tensorboardX
    
    cd ${cwd}
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Copy model files to fairseq..."
    rsync -a contentvec/ fairseq/fairseq/
    
fi
