#!/usr/bin/env bash
set -euo pipefail

# Train against an already-running `swift rollout` server.
# Start run_swift_rollout_server.sh in another terminal first.

RLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The installed `wandb` module comes from corporate byted-wandb 0.13.x.
# Disable its unreliable service process and run the legacy backend in-thread.
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
unset WANDB_REQUIRE_SERVICE
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,3}"
NUM_GPUS="$(awk -F',' '{print NF}' <<<"$TRAIN_GPUS")"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="${VLLM_URL:-http://${VLLM_HOST}:${VLLM_PORT}}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-51216}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-900}"

USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-8}"

PROJECT_NAME="${PROJECT_NAME:-rllm-swift}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepscaler-qwen3-4b-server}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate)}"

NCCL_LIB_DIR="$(
    "$PYTHON_BIN" -c \
        'import importlib.util, pathlib; spec = importlib.util.find_spec("nvidia.nccl"); print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "lib" if spec else "")'
)"
if [[ -n "$NCCL_LIB_DIR" && -f "$NCCL_LIB_DIR/libnccl.so.2" ]]; then
    export LD_LIBRARY_PATH="$NCCL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

LOG_DIR="${LOG_DIR:-$RLLM_ROOT/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/rllm-swift-deepscaler-qwen3-4b-server}"
LAUNCHER_LOG="$LOG_DIR/${EXPERIMENT_NAME}_launcher_${RUN_TS}.log"
TRAIN_OUT="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.out"
TRAIN_ERR="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.err"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
ln -sfn "$(basename "$LAUNCHER_LOG")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.launcher.log"
ln -sfn "$(basename "$TRAIN_OUT")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.out"
ln -sfn "$(basename "$TRAIN_ERR")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.err"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

cd "$RLLM_ROOT"

echo "==== Checking Swift rollout server ===="
"$PYTHON_BIN" - "$VLLM_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")

def get(path):
    with urllib.request.urlopen(base_url + path, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return json.loads(response.read())

try:
    health = get("/health/")
    world_size = get("/get_world_size/")
except Exception as exc:
    raise SystemExit(
        f"Swift rollout server is not ready at {base_url}: {exc}\n"
        "Start ./run_swift_rollout_server.sh first and wait for "
        "'Application startup complete'."
    )

print(f"health={health}")
print(f"world_size={world_size}")
PY

echo "==== Training environment preflight ===="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON_BIN" - "$NUM_GPUS" <<'PY'
import ctypes
import sys

import torch
import wandb

required_gpus = int(sys.argv[1])
nccl = ctypes.CDLL("libnccl.so.2")
nccl_version = ctypes.c_int()
if nccl.ncclGetVersion(ctypes.byref(nccl_version)) != 0:
    raise SystemExit("Failed to query the loaded NCCL library version.")

torch_nccl = torch.cuda.nccl.version()
expected_nccl = torch_nccl[0] * 10_000 + torch_nccl[1] * 100 + torch_nccl[2]
print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_nccl={torch_nccl}")
print(f"loaded_nccl={nccl_version.value}")
print(f"visible_gpu_count={torch.cuda.device_count()}")
print(f"wandb_provider=byted-wandb")
print(f"wandb_module_version={getattr(wandb, '__version__', 'unknown')}")

if nccl_version.value != expected_nccl:
    raise SystemExit(
        f"NCCL mismatch: PyTorch expects {expected_nccl}, "
        f"but libnccl.so.2 reports {nccl_version.value}."
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the training process.")
if torch.cuda.device_count() != required_gpus:
    raise SystemExit(
        f"Expected {required_gpus} training GPUs, "
        f"but {torch.cuda.device_count()} are visible."
    )
if not hasattr(wandb, "init"):
    raise SystemExit("Corporate byted-wandb is not installed correctly.")
PY

USE_LORA_BOOL=false
if [[ "$USE_LORA" == "1" ]]; then
    USE_LORA_BOOL=true
fi

echo "==== DeepScaler Swift training ===="
echo "train_gpus=$TRAIN_GPUS (num_processes=$NUM_GPUS)"
echo "vllm_url=$VLLM_URL"
echo "vllm_group_port=$VLLM_GROUP_PORT"
echo "model_path=$MODEL_PATH"
echo "use_lora=$USE_LORA (rank=$LORA_RANK alpha=$LORA_ALPHA)"
echo "max_prompt_length=$MAX_PROMPT_LENGTH"
echo "max_response_length=$MAX_RESPONSE_LENGTH"
echo "train_batch_size_per_gpu=$TRAIN_BATCH_SIZE"
echo "num_minibatches=$NUM_MINIBATCHES"
echo "launcher_log=$LAUNCHER_LOG"
echo "train_out=$TRAIN_OUT"
echo "train_err=$TRAIN_ERR"
echo "checkpoint_dir=$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$ACCELERATE_BIN" launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m examples.deepscaler.train_deepscaler_swift \
    model.name="$MODEL_PATH" \
    model.trust_remote_code=true \
    model.use_lora="$USE_LORA_BOOL" \
    model.lora_rank="$LORA_RANK" \
    model.lora_alpha="$LORA_ALPHA" \
    rollout.mode=server \
    rollout.base_url="$VLLM_URL" \
    rollout.group_port="$VLLM_GROUP_PORT" \
    rollout.sync_weights=true \
    rollout.weight_sync_mode=auto \
    rollout.server_timeout_s="$SERVER_TIMEOUT_S" \
    rollout.batch_size=16 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=8 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    training.group_size=8 \
    training.num_minibatches="$NUM_MINIBATCHES" \
    training.learning_rate=1e-6 \
    sampling.temperature=0.6 \
    sampling.top_p=0.95 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    agent.max_steps=1 \
    trainer.total_epochs=1 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=true \
    trainer.save_freq=20000 \
    trainer.test_freq=20 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    > >(tee -a "$TRAIN_OUT") \
    2> >(tee -a "$TRAIN_ERR" >&2)
