#!/bin/bash
#SBATCH --job-name=qwen3_maml_plan+sft_sci_eval
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o ../logs/sciworld_eval/test.out
#SBATCH -e ../logs/sciworld_eval/test.err
module add anaconda3/2023.3
module add cuda/12.9
module add jdk/11
cd /mnt/home/user28/MPRL/rllm
source .venv/bin/activate
cd ..

python -m maml.run_sciworld_eval \
    --config ./maml/configs/sciworld_eval_config.yaml \
    --test_idx_path ./data/eval_idx/sciworld/test_indices.json 
