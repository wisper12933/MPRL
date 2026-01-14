#!/bin/bash
#SBATCH --job-name=qwen3_MAML_plan+sft_web_eval
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/metaplan_eval/qwen3_MAML_plan+sft_web_eval.out
#SBATCH -e ../logs/metaplan_eval/qwen3_MAML_plan+sft_web_eval.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL/rllm
source .venv/bin/activate
cd ..

python -m maml.run_metaplan_eval --config ./maml/configs/metaplan_eval_config.yaml