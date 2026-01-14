#!/bin/bash
#SBATCH --job-name=qwen3_MAML
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH -o ../logs/training/test.out
#SBATCH -e ../logs/training/test.err
#SBATCH --nodelist=L20004
# 节点选择L20004或L20007，其他的会有问题
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL/rllm
source .venv/bin/activate
cd ..

python -m maml.run_training --config ./maml/configs/training_config.yaml