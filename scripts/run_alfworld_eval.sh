#!/bin/bash
#SBATCH --job-name=qwen3_sft-task_alf1
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/alfworld_eval/test.out
#SBATCH -e ../logs/alfworld_eval/test.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL/rllm
source .venv/bin/activate
cd ..
# qwen3_sft-task_alf  qwen3_sft_alf
export ALFWORLD_DATA='/mnt/home/user28/alfworld_data'
python -m maml.run_alfworld_eval --config ./maml/configs/alfworld_eval_config.yaml