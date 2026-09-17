#!/usr/bin/env bash
set -euo pipefail

# Terminal 1: start a LoRA-capable Swift rollout server and keep it running.
# The base weights are loaded from disk; the trainer synchronizes the selected
# task adapter before the first rollout.

RLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$RLLM_ROOT/.." && pwd)"
TASK="${TASK:-webshop}"
BASE_MODEL="${BASE_MODEL:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/hdfs/lixingzuo/qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora}"

case "$TASK" in
    webshop) ADAPTER_DIRNAME="Qwen3-4B-Instruct-MAML-plan-sft-web" ;;
    alfworld) ADAPTER_DIRNAME="Qwen3-4B-Instruct-MAML-plan-sft-alf" ;;
    sciworld) ADAPTER_DIRNAME="Qwen3-4B-Instruct-MAML-plan-sft-sci" ;;
    *) echo "Unsupported TASK=$TASK (use webshop|alfworld|sciworld)" >&2; exit 2 ;;
esac

ADAPTER_PATH="${ADAPTER_PATH:-$ADAPTER_ROOT/$ADAPTER_DIRNAME}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-mprl311/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-$REPO_ROOT/.venv-mprl311/bin/swift}"
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_TP_SIZE="$(awk -F',' '{print NF}' <<<"$VLLM_GPUS")"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"

if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    echo "Adapter config not found: $ADAPTER_PATH/adapter_config.json" >&2
    exit 1
fi
LORA_RANK="$("$PYTHON_BIN" - "$ADAPTER_PATH/adapter_config.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["r"])
PY
)"

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
    echo "Port $VLLM_PORT is already in use. Stop the old server or choose another VLLM_PORT." >&2
    command -v ss >/dev/null && ss -lptn "sport = :$VLLM_PORT" || true
    exit 1
fi

RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$RLLM_ROOT/logs/mprl/$TASK}"
VLLM_LOG="$LOG_DIR/vllm_${RUN_TS}.log"
mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$VLLM_LOG")" "$LOG_DIR/latest.vllm.log"
exec > >(tee -a "$VLLM_LOG") 2>&1

export PYTHONPATH="$RLLM_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
cd "$RLLM_ROOT"

echo "==== MPRL Swift rollout server ===="
echo "task=$TASK"
echo "base_model=$BASE_MODEL"
echo "initial_adapter_source=$ADAPTER_PATH"
echo "lora_rank=$LORA_RANK"
echo "vllm_gpus=$VLLM_GPUS"
echo "vllm_url=http://${VLLM_HOST}:${VLLM_PORT}"
echo "max_model_len=$MAX_MODEL_LEN"
echo "vllm_log=$VLLM_LOG"
echo
echo "Wait for 'Application startup complete', then verify:"
echo "  curl -f http://${VLLM_HOST}:${VLLM_PORT}/health/"
echo

exec env CUDA_VISIBLE_DEVICES="$VLLM_GPUS" "$SWIFT_BIN" rollout \
    --model "$BASE_MODEL" \
    --model_type qwen3 \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --torch_dtype bfloat16 \
    --vllm_tensor_parallel_size "$VLLM_TP_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_len "$MAX_MODEL_LEN" \
    --vllm_enable_lora true \
    --vllm_max_lora_rank "$LORA_RANK" \
    --vllm_engine_kwargs '{"load_format":"auto"}'
