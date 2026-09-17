#!/usr/bin/env bash
set -euo pipefail

# Start the official ms-swift rollout server in the foreground.
# Keep this terminal running while training uses the server from another terminal.

RLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen3}"
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_TP_SIZE="$(awk -F',' '{print NF}' <<<"$VLLM_GPUS")"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-32}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepscaler-qwen3-4b-server}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
SWIFT_BIN="${SWIFT_BIN:-$(command -v swift)}"
LOG_DIR="${LOG_DIR:-$RLLM_ROOT/logs}"
VLLM_LOG="$LOG_DIR/${EXPERIMENT_NAME}_vllm_${RUN_TS}.log"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$VLLM_LOG")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.vllm.log"
exec > >(tee -a "$VLLM_LOG") 2>&1

cd "$RLLM_ROOT"

if ! "$PYTHON_BIN" - "$VLLM_PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
then
    echo "Port $VLLM_PORT is already in use. Stop the old server or set another VLLM_PORT."
    command -v ss >/dev/null && ss -lptn "sport = :$VLLM_PORT" || true
    exit 1
fi

LORA_ARGS=()
if [[ "$USE_LORA" == "1" ]]; then
    LORA_ARGS=(
        --vllm_enable_lora true
        --vllm_max_lora_rank "$LORA_RANK"
    )
fi

echo "==== Starting Swift rollout server ===="
echo "model_path=$MODEL_PATH"
echo "model_type=$MODEL_TYPE"
echo "vllm_gpus=$VLLM_GPUS (tensor_parallel_size=$VLLM_TP_SIZE)"
echo "vllm_url=http://${VLLM_HOST}:${VLLM_PORT}"
echo "max_model_len=$MAX_MODEL_LEN"
echo "vllm_log=$VLLM_LOG"
echo
echo "Wait for 'Application startup complete', then run:"
echo "  curl -f http://${VLLM_HOST}:${VLLM_PORT}/health/"
echo

exec env CUDA_VISIBLE_DEVICES="$VLLM_GPUS" "$SWIFT_BIN" rollout \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --torch_dtype bfloat16 \
    --vllm_tensor_parallel_size "$VLLM_TP_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_len "$MAX_MODEL_LEN" \
    "${LORA_ARGS[@]}"
