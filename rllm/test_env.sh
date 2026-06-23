#!/bin/bash
#SBATCH --job-name=rllm_demo
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/test_env.out
#SBATCH -e ./logs/test_env.err

# 1. Load modules and set up environment
module load anaconda3/2023.3
module load cuda/12.4

# Change directory to the workspace
cd /mnt/home/user28/MPRL/rllm

# Activate the virtual environment
source activate rllm

python test_env.py