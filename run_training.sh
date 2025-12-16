#!/bin/bash
#SBATCH --job-name=qwen2.5_gen_MAML
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH -o ./logs/training/qwen2.5_gen_MAML.out
#SBATCH -e ./logs/training/qwen2.5_gen_MAML.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL
source activate mprl

python -m maml.run_training --config ./configs/training_config.yaml