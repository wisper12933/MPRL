#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RLLM_ROOT="$REPO_ROOT/rllm"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-mprl311/bin/python}"
TASK="${TASK:-webshop}"
BASE_MODEL="${BASE_MODEL:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}/v1"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LIMIT="${LIMIT:-8}"
REPEAT_K="${REPEAT_K:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
READY_RETRIES="${READY_RETRIES:-120}"
READY_SLEEP_SECONDS="${READY_SLEEP_SECONDS:-5}"

case "$TASK" in
    webshop)
        DEFAULT_LORA_NAME="webshop-lora-adapter"
        DEFAULT_LORA_PATH="/mnt/hdfs/lixingzuo/qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora/Qwen3-4B-Instruct-MAML-plan-sft-web"
        DEFAULT_N_PARALLEL_AGENTS="2"
        DEFAULT_MAX_STEPS="12"
        ;;
    alfworld)
        DEFAULT_LORA_NAME="alfworld-lora-adapter"
        DEFAULT_LORA_PATH="/mnt/hdfs/lixingzuo/qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora/Qwen3-4B-Instruct-MAML-plan-sft-alf"
        DEFAULT_N_PARALLEL_AGENTS="2"
        DEFAULT_MAX_STEPS="40"
        ;;
    sciworld)
        DEFAULT_LORA_NAME="sciworld-lora-adapter"
        DEFAULT_LORA_PATH="/mnt/hdfs/lixingzuo/qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora/Qwen3-4B-Instruct-MAML-plan-sft-sci"
        DEFAULT_N_PARALLEL_AGENTS="1"
        DEFAULT_MAX_STEPS="60"
        ;;
    *)
        echo "Unsupported TASK: $TASK" >&2
        exit 1
        ;;
esac

LORA_NAME="${LORA_NAME:-$DEFAULT_LORA_NAME}"
LORA_PATH="${LORA_PATH:-$DEFAULT_LORA_PATH}"
N_PARALLEL_AGENTS="${N_PARALLEL_AGENTS:-$DEFAULT_N_PARALLEL_AGENTS}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$RLLM_ROOT/logs/interact/$TASK"
LAUNCHER_LOG="$LOG_DIR/launcher_${RUN_TS}.log"
VLLM_LOG="$LOG_DIR/vllm_${RUN_TS}.log"
INTERACT_LOG="$LOG_DIR/interact_${RUN_TS}.log"
mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LAUNCHER_LOG")" "$LOG_DIR/latest.launcher.log"
ln -sfn "$(basename "$VLLM_LOG")" "$LOG_DIR/latest.vllm.log"
ln -sfn "$(basename "$INTERACT_LOG")" "$LOG_DIR/latest.interact.log"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$BASE_MODEL" && ! -f "$BASE_MODEL/config.json" ]]; then
    echo "Base model path does not exist: $BASE_MODEL" >&2
    exit 1
fi

if [[ ! -d "$LORA_PATH" ]]; then
    echo "LoRA path does not exist: $LORA_PATH" >&2
    exit 1
fi

export PYTHONPATH="$RLLM_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld_data}"
RLLM_JAVA_HOME="${RLLM_JAVA_HOME:-/opt/tiger/jdk/jdk11}"
RLLM_JDK_HOME="${RLLM_JDK_HOME:-$RLLM_JAVA_HOME}"
RLLM_JRE_HOME="${RLLM_JRE_HOME:-$RLLM_JAVA_HOME}"
RLLM_JVM_PATH="${RLLM_JVM_PATH:-$RLLM_JAVA_HOME/lib/server/libjvm.so}"
export JAVA_HOME="$RLLM_JAVA_HOME"
export JDK_HOME="$RLLM_JDK_HOME"
export JRE_HOME="$RLLM_JRE_HOME"
export JVM_PATH="$RLLM_JVM_PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export PATH="$JAVA_HOME/bin:$PATH"

if [[ ! -x "$JAVA_HOME/bin/java" ]]; then
    echo "Java runtime not found: $JAVA_HOME/bin/java" >&2
    exit 1
fi

if [[ ! -f "$JVM_PATH" ]]; then
    echo "Java libjvm not found: $JVM_PATH" >&2
    exit 1
fi

VLLM_PID=""
INTERACT_PID=""
cleanup() {
    local code="${1:-0}"
    if [[ -n "$INTERACT_PID" ]] && kill -0 "$INTERACT_PID" 2>/dev/null; then
        kill -TERM "$INTERACT_PID" 2>/dev/null || true
        wait "$INTERACT_PID" 2>/dev/null || true
    fi
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    exit "$code"
}
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

cd "$RLLM_ROOT"

echo "==== RL sampling launcher ===="
echo "task=$TASK"
echo "python_bin=$PYTHON_BIN"
echo "base_model=$BASE_MODEL"
echo "lora_name=$LORA_NAME"
echo "lora_path=$LORA_PATH"
echo "base_url=$BASE_URL"
echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "max_model_len=$MAX_MODEL_LEN"
echo "max_num_seqs=$MAX_NUM_SEQS"
echo "limit=$LIMIT"
echo "repeat_k=$REPEAT_K"
echo "n_parallel_agents=$N_PARALLEL_AGENTS"
echo "max_steps=$MAX_STEPS"
echo "max_prompt_length=$MAX_PROMPT_LENGTH"
echo "max_response_length=$MAX_RESPONSE_LENGTH"
echo "temperature=$TEMPERATURE"
echo "top_p=$TOP_P"
echo "launcher_log=$LAUNCHER_LOG"
echo "vllm_log=$VLLM_LOG"
echo "interact_log=$INTERACT_LOG"
"$JAVA_HOME/bin/java" -version

echo "Starting vLLM server..."
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enable-lora \
    --lora-modules "$LORA_NAME=$LORA_PATH" \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM readiness on $BASE_URL/models ..."
retry_count=0
while [[ "$retry_count" -lt "$READY_RETRIES" ]]; do
    response="$(curl -s "$BASE_URL/models" 2>/dev/null || true)"
    if [[ -n "$response" ]] && grep -q '"data"' <<<"$response"; then
        echo "vLLM is ready. /models response:"
        echo "$response"
        break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready. See $VLLM_LOG" >&2
        cleanup 1
    fi

    retry_count=$((retry_count + 1))
    echo "Waiting... (${retry_count}/${READY_RETRIES})"
    sleep "$READY_SLEEP_SECONDS"
done

if [[ "$retry_count" -eq "$READY_RETRIES" ]]; then
    echo "Timed out waiting for vLLM readiness. See $VLLM_LOG" >&2
    cleanup 1
fi

echo "Starting interact sampler..."
"$PYTHON_BIN" -m mprl.run_interact \
    --task "$TASK" \
    --base-url "$BASE_URL" \
    --base-model "$BASE_MODEL" \
    --model-alias "$LORA_NAME" \
    --limit "$LIMIT" \
    --repeat-k "$REPEAT_K" \
    --n-parallel-agents "$N_PARALLEL_AGENTS" \
    --max-steps "$MAX_STEPS" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    >"$INTERACT_LOG" 2>&1 &
INTERACT_PID=$!
wait "$INTERACT_PID"
INTERACT_EXIT_CODE=$?

cleanup "$INTERACT_EXIT_CODE"
