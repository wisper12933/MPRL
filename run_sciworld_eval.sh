#!/bin/bash
#SBATCH --job-name=llama3.1_maml_plan_sft_sci_eval
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/sciworld_eval/llama3.1_maml-plan+sft-task_sci.out
#SBATCH -e ./logs/sciworld_eval/llama3.1_maml-plan+sft-task_sci.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/MPRL
source activate mprl

export PYTHONPATH=$PYTHONPATH:.
python -m maml.run_sciworld_eval \
    --config ./configs/sciworld_eval_config.yaml \
    --test_idx_path ./data/sciworld/test_indices.json 
