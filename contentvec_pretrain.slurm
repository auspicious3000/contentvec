#!/bin/bash
#SBATCH -J contentvec_pretrain
#SBATCH -o contentvec_pretrain_%j.out
#SBATCH -e contentvec_pretrain_%j.err
#SBATCH --time=24
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:4
#SBATCH --nodes=8
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=
CONDA_ROOT=

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

srun ./run_pretrain_server.sh

echo "Run completed at:- "
date
