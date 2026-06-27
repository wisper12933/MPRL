#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RLLM_ROOT="$REPO_ROOT/rllm"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-16k}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$RLLM_ROOT/logs"
LAUNCHER_LOG="$LOG_DIR/${EXPERIMENT_NAME}_launcher_${RUN_TS}.log"
TRAIN_OUT="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.out"
TRAIN_ERR="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.err"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LAUNCHER_LOG")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.launcher.log"
ln -sfn "$(basename "$TRAIN_OUT")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.out"
ln -sfn "$(basename "$TRAIN_ERR")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.err"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

cd "$RLLM_ROOT"

echo "==== Direct RL training launcher ===="
echo "python_bin=$PYTHON_BIN"
echo "model_path=$MODEL_PATH"
echo "experiment_name=$EXPERIMENT_NAME"
echo "launcher_log=$LAUNCHER_LOG"
echo "train_out=$TRAIN_OUT"
echo "train_err=$TRAIN_ERR"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

"$PYTHON_BIN" -m examples.deepscaler.train_deepscaler \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=30 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
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
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=600 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=1 \
    >"$TRAIN_OUT" 2>"$TRAIN_ERR"
