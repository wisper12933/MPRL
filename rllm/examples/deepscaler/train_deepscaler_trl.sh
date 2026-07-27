#!/usr/bin/env bash
set -euo pipefail

RLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepscaler-trl-smoke}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$RLLM_ROOT/logs"
LAUNCHER_LOG="$LOG_DIR/${EXPERIMENT_NAME}_launcher_${RUN_TS}.log"
TRAIN_OUT="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.out"
TRAIN_ERR="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.err"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LAUNCHER_LOG")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.launcher.log"
ln -sfn "$(basename "$TRAIN_OUT")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.out"
ln -sfn "$(basename "$TRAIN_ERR")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.err"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

cd "$RLLM_ROOT"

echo "==== TRL DeepScaler training launcher ===="
echo "python_bin=$PYTHON_BIN"
echo "model_path=$MODEL_PATH"
echo "experiment_name=$EXPERIMENT_NAME"
echo "launcher_log=$LAUNCHER_LOG"
echo "train_out=$TRAIN_OUT"
echo "train_err=$TRAIN_ERR"

"$PYTHON_BIN" -m examples.deepscaler.train_deepscaler_trl \
    model.name="$MODEL_PATH" \
    data.train_batch_size=2 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    training.group_size=8 \
    training.learning_rate=1e-6 \
    training.num_minibatches=1 \
    sampling.temperature=0.6 \
    sampling.top_p=0.95 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    agent.max_steps=1 \
    trainer.total_epochs=1 \
    trainer.logger=['console'] \
    trainer.project_name='rllm-trl' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir="outputs/${EXPERIMENT_NAME}_checkpoints" \
    >"$TRAIN_OUT" 2>"$TRAIN_ERR"
