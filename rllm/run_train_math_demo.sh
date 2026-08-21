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

"$PYTHON_BIN" -m examples.deepscaler.train_deepscaler_tinker \
    model.name="$MODEL_PATH" \
    algorithm.adv_estimator=grpo \
    training.learning_rate=1e-6 \
    training.group_size=8 \
    sampling.temperature=0.6 \
    sampling.top_p=0.95 \
    data.train_batch_size=32 \
    data.val_batch_size=30 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    agent.max_steps=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.save_freq=2000 \
    trainer.test_freq=10 \
    trainer.default_local_dir="$LOG_DIR/${EXPERIMENT_NAME}_checkpoints" \
    trainer.total_epochs=1 \
    >"$TRAIN_OUT" 2>"$TRAIN_ERR"
