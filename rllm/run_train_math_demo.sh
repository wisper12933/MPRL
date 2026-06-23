#!/bin/bash
#SBATCH --job-name=rllm_demo
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:4
#SBATCH -o ./logs/rllm_demo/train_math.out
#SBATCH -e ./logs/rllm_demo/train_math.err

# ============ 1. 加载宿主机基础模块 ============
module load singularity/4.3.5
module load cuda/12.9 
module load anaconda3/2023.3  # 加载 anaconda 模块

# 动态获取宿主机 CUDA 12.9 与 conda 的安装路径
HOST_CUDA_PATH=$(dirname $(dirname $(which nvcc)))
HOST_CONDA_PATH=$(dirname $(dirname $(which conda)))

echo "Host CUDA: $HOST_CUDA_PATH"
echo "Host Conda: $HOST_CONDA_PATH"

# ============ 2. 准备个人缓存目录 (解决 Permission denied) ============
MY_BASE_DIR="/mnt/home/user28"
MY_CACHE_DIR="$MY_BASE_DIR/.cache/rllm_runtime"
mkdir -p "$MY_CACHE_DIR/singularity" \
         "$MY_CACHE_DIR/singularity_tmp" \
         "$MY_CACHE_DIR/huggingface" \
         "$MY_CACHE_DIR/flashinfer" \
         "$MY_CACHE_DIR/triton" \
         "$MY_CACHE_DIR/xdg_config" \
         "$MY_CACHE_DIR/pycache" \
         "$MY_CACHE_DIR/xdg_cache" \
         "$MY_BASE_DIR/.cache/torch" \
         "$MY_BASE_DIR/.cache/huggingface" \

# 设置 Singularity 运行缓存（容器外使用）
export SINGULARITY_CACHEDIR="$MY_CACHE_DIR/singularity"
export SINGULARITY_TMPDIR="$MY_CACHE_DIR/singularity_tmp"

# ============ 3. 定义容器内环境变量 (SINGULARITYENV_ 前缀) ============
# 这些变量会自动注入容器
export SINGULARITYENV_HF_HOME="$MY_BASE_DIR/.cache/huggingface"
export SINGULARITYENV_TORCH_HOME="$MY_BASE_DIR/.cache/torch"
export SINGULARITYENV_FLASHINFER_WORKSPACE_DIR="$MY_CACHE_DIR/flashinfer"
export SINGULARITYENV_TRITON_CACHE_DIR="$MY_CACHE_DIR/triton"
export SINGULARITYENV_XDG_CONFIG_HOME="$MY_CACHE_DIR/xdg_config"
export SINGULARITYENV_PYTHONPYCACHEPREFIX="$MY_CACHE_DIR/pycache"
export SINGULARITYENV_XDG_CACHE_HOME="$MY_CACHE_DIR/xdg_cache"

# 强制注入 CUDA 和 Conda 路径到容器 PATH
export SINGULARITYENV_CUDA_HOME="$HOST_CUDA_PATH"
export SINGULARITYENV_PATH="$HOST_CONDA_PATH/bin:$HOST_CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export SINGULARITYENV_LD_LIBRARY_PATH="$HOST_CUDA_PATH/lib64:$HOST_CUDA_PATH/extras/CUPTI/lib64"

# ============ 4. 运行容器 ============
SANDBOX_CONTAINER="/mnt/home/user28/ubuntu22.04"
singularity shell --nv --cleanenv \
    --home $MY_BASE_DIR \
    --bind $MY_BASE_DIR:$MY_BASE_DIR \
    --bind $MY_CACHE_DIR:$MY_CACHE_DIR \
    --bind /tmp:/tmp \
    --bind $HOST_CUDA_PATH:$HOST_CUDA_PATH \
    --bind $HOST_CONDA_PATH:$HOST_CONDA_PATH \
    $SANDBOX_CONTAINER << 'EOF'
export PATH=/mnt/data/hpc/support/soft/cuda/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/hpc/support/soft/cuda/cuda-12.9/lib64:$LD_LIBRARY_PATH

source /mnt/data/hpc/support/soft/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/home/user28/.conda/envs/rllm

echo '=== 容器内环境检查 ==='
echo 'GCC Version:' $(gcc --version | head -n 1)
echo 'NVCC Version:' $(nvcc --version | grep release)
echo 'Python path:' $(which python3)
echo 'VLLM Stats Disabled:' \$VLLM_NO_USAGE_STATS

python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('is_available', torch.cuda.is_available())"

# 设置训练相关优化参数
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export NCCL_P2P_DISABLE=1

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export WANDB_API_KEY=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
export WANDB_MODE=offline

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

MODEL_PATH=/mnt/home/user28/llms/Qwen3-4B-Instruct

python3 -m examples.deepscaler.train_deepscaler \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=30 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='qwen3-4b-16k' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=400 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=1

EOF