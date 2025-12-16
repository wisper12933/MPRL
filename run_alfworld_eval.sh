#!/bin/bash
#SBATCH --job-name=llama3.1_plan_sft_alf_eval
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/alfworld_eval/llama3.1_maml-plan+sft-task_alf.out
#SBATCH -e ./logs/alfworld_eval/llama3.1_maml-plan+sft-task_alf.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL
source activate mprl

export PYTHONPATH=$PYTHONPATH:.
export ALFWORLD_DATA='/mnt/home/user28/alfworld_data'
python -m maml.run_alfworld_eval --config ./configs/alfworld_eval_config.yaml