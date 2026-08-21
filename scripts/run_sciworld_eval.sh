#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv-mprl311/bin/python"
TEST_IDX_PATH="$REPO_ROOT/data/indices/sciworld/test_indices.json"
JAVA_HOME="${JAVA_HOME:-/opt/tiger/jdk/jdk11}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/sciworld_eval_${RUN_TS}.log"
LATEST_LOG="$LOG_DIR/sciworld_eval.latest.log"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"
exec >"$LOG_FILE" 2>&1

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -x "$JAVA_HOME/bin/java" ]]; then
    echo "Java 11 runtime not found: $JAVA_HOME/bin/java" >&2
    exit 1
fi

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m maml.run_sciworld_eval \
    --config ./maml/configs/sciworld_eval_config.yaml \
    --test_idx_path "$TEST_IDX_PATH"
