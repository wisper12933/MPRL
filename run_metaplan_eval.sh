#!/bin/bash
#SBATCH --job-name=qwen_sft_plan_sft_sci_eval
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/metaplan_eval/qwen2.5_sft_plan+sft_sci_eval.out
#SBATCH -e ./logs/metaplan_eval/qwen2.5_sft_plan+sft_sci_eval.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL
source activate mprl

python -m maml.run_metaplan_eval --config ./configs/metaplan_eval_config.yaml