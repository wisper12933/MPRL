#!/bin/bash
#SBATCH --job-name=qwen3_sft-web
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/webshop_eval/qwen3_sft-task_web.out
#SBATCH -e ../logs/webshop_eval/qwen3_sft-task_web.err
module add anaconda3/2023.3
module add cuda/12.9
module add jdk/11
cd /mnt/home/user28/MPRL/rllm
source .venv/bin/activate
cd ..

python -m maml.run_webshop_eval \
    --config ./maml/configs/webshop_eval_config.yaml \
    --test_idx_path ./data/eval_idx/webshop/test_indices.json 
