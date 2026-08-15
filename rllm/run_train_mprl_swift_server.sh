#!/usr/bin/env bash
set -euo pipefail

# Terminal 2: train one MPRL task against an already healthy Swift server.

RLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$RLLM_ROOT/.." && pwd)"
TASK="${TASK:-webshop}"
BASE_MODEL="${BASE_MODEL:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/hdfs/lixingzuo/qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora}"

case "$TASK" in
    webshop)
        ADAPTER_DIRNAME="Qwen3-4B-Instruct-MAML-plan-sft-web"
        DEFAULT_MAX_STEPS=12
        DEFAULT_N_PARALLEL=2
        ;;
    alfworld)
        ADAPTER_DIRNAME="Qwen3-4B-Instruct-MAML-plan-sft-alf"
        DEFAULT_MAX_STEPS=40
        DEFAULT_N_PARALLEL=2
        ;;
    sciworld)
        ADAPTER_DIRNAME="Qwen3-4B-Instruct-MAML-plan-sft-sci"
        DEFAULT_MAX_STEPS=60
        DEFAULT_N_PARALLEL=1
        ;;
    *) echo "Unsupported TASK=$TASK (use webshop|alfworld|sciworld)" >&2; exit 2 ;;
esac

ADAPTER_PATH="${ADAPTER_PATH:-$ADAPTER_ROOT/$ADAPTER_DIRNAME}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,3}"
NUM_GPUS="$(awk -F',' '{print NF}' <<<"$TRAIN_GPUS")"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="${VLLM_URL:-http://${VLLM_HOST}:${VLLM_PORT}}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-51216}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-900}"

MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
N_PARALLEL_AGENTS="${N_PARALLEL_AGENTS:-$DEFAULT_N_PARALLEL}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
VAL_LIMIT="${VAL_LIMIT:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
PLANNING_MAX_TOKENS="${PLANNING_MAX_TOKENS:-1024}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"

PROJECT_NAME="${PROJECT_NAME:-rllm-mprl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mprl-${TASK}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/rllm-mprl-${TASK}}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-mprl311/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$REPO_ROOT/.venv-mprl311/bin/accelerate}"

export PYTHONPATH="$RLLM_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld_data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# This is the corporate byted-wandb package. Its legacy in-thread backend is
# reliable on these shared nodes and still uploads to the internal service.
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
unset WANDB_REQUIRE_SERVICE
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

JAVA_HOME="${JAVA_HOME:-/opt/tiger/jdk/jdk11}"
export JAVA_HOME JDK_HOME="${JDK_HOME:-$JAVA_HOME}" JRE_HOME="${JRE_HOME:-$JAVA_HOME}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"

NCCL_LIB_DIR="$(
    "$PYTHON_BIN" -c \
        'import importlib.util, pathlib; spec = importlib.util.find_spec("nvidia.nccl"); print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "lib" if spec else "")'
)"
if [[ -n "$NCCL_LIB_DIR" && -f "$NCCL_LIB_DIR/libnccl.so.2" ]]; then
    export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$LD_LIBRARY_PATH"
fi

if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    echo "Adapter not found: $ADAPTER_PATH" >&2
    exit 1
fi

echo "==== Checking existing Swift rollout server ===="
"$PYTHON_BIN" - "$VLLM_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
for path in ("/health/", "/get_world_size/"):
    try:
        with urllib.request.urlopen(base_url + path, timeout=10) as response:
            payload = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            print(f"{path} -> {payload}")
    except Exception as exc:
        raise SystemExit(
            f"Swift rollout is not ready at {base_url}: {exc}\n"
            "Run ./run_mprl_swift_rollout_server.sh first and wait for "
            "'Application startup complete'."
        )
PY

RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$RLLM_ROOT/logs/mprl/$TASK}"
LAUNCHER_LOG="$LOG_DIR/train_launcher_${RUN_TS}.log"
TRAIN_OUT="$LOG_DIR/train_${RUN_TS}.out"
TRAIN_ERR="$LOG_DIR/train_${RUN_TS}.err"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
ln -sfn "$(basename "$LAUNCHER_LOG")" "$LOG_DIR/latest.train.launcher.log"
ln -sfn "$(basename "$TRAIN_OUT")" "$LOG_DIR/latest.train.out"
ln -sfn "$(basename "$TRAIN_ERR")" "$LOG_DIR/latest.train.err"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

cd "$RLLM_ROOT"
echo "==== MPRL Swift training ===="
echo "task=$TASK"
echo "base_model=$BASE_MODEL"
echo "adapter_path=$ADAPTER_PATH"
echo "train_gpus=$TRAIN_GPUS"
echo "vllm_url=$VLLM_URL"
echo "max_steps=$MAX_STEPS"
echo "n_parallel_agents=$N_PARALLEL_AGENTS"
echo "train_limit=$TRAIN_LIMIT val_limit=$VAL_LIMIT"
echo "checkpoint_dir=$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$ACCELERATE_BIN" launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m mprl.train_interact \
    mprl.task="$TASK" \
    mprl.train_limit="$TRAIN_LIMIT" \
    mprl.test_limit="$VAL_LIMIT" \
    mprl.alfworld_data="$ALFWORLD_DATA" \
    mprl.max_steps="$MAX_STEPS" \
    mprl.n_parallel_agents="$N_PARALLEL_AGENTS" \
    model.name="$BASE_MODEL" \
    model.adapter_path="$ADAPTER_PATH" \
    rollout.base_url="$VLLM_URL" \
    rollout.group_port="$VLLM_GROUP_PORT" \
    rollout.server_timeout_s="$SERVER_TIMEOUT_S" \
    rollout.sync_weights=true \
    rollout.weight_sync_mode=auto \
    planning.max_tokens="$PLANNING_MAX_TOKENS" \
    training.group_size="$GROUP_SIZE" \
    training.num_minibatches="$NUM_MINIBATCHES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    'trainer.logger=[console,wandb]' \
    > >(tee -a "$TRAIN_OUT") \
    2> >(tee -a "$TRAIN_ERR" >&2)
