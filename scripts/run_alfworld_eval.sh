#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv-mprl311/bin/python"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/alfworld_eval_${RUN_TS}.log"
LATEST_LOG="$LOG_DIR/alfworld_eval.latest.log"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"
exec >"$LOG_FILE" 2>&1

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

cd "$REPO_ROOT"

# qwen3_sft-task_alf  qwen3_sft_alf
export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/data/alfworld_data}"
"$PYTHON_BIN" -m maml.run_alfworld_eval --config ./maml/configs/alfworld_eval_config.yaml