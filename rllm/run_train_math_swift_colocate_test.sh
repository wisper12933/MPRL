#!/usr/bin/env bash
set -euo pipefail

RLLM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# byted-wandb 0.13.x requires its service subprocess on import, and that
# subprocess can time out while claiming a port on busy worker nodes. This flag
# is the only supported opt-out; the legacy backend still uploads runs.
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
unset WANDB_REQUIRE_SERVICE
# The legacy backend otherwise spawns a fresh interpreter that re-imports wandb
# from the shared filesystem, which regularly exceeds the 60s init timeout here.
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct}"
NUM_GPUS="${NUM_GPUS:-4}"
PROJECT_NAME="${PROJECT_NAME:-rllm-swift}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepscaler-qwen3-4b-colocate}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate)}"
NCCL_LIB_DIR="$(
    "$PYTHON_BIN" -c \
        'import importlib.util, pathlib; spec = importlib.util.find_spec("nvidia.nccl"); print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "lib" if spec else "")'
)"

# Prefer the virtual-environment NCCL and verify below that the shared library
# actually matches the NCCL version against which PyTorch was built.
if [[ -n "$NCCL_LIB_DIR" && -f "$NCCL_LIB_DIR/libnccl.so.2" ]]; then
    export LD_LIBRARY_PATH="$NCCL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

LOG_DIR="${LOG_DIR:-$RLLM_ROOT/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/rllm-swift-deepscaler-qwen3-4b}"
LAUNCHER_LOG="$LOG_DIR/${EXPERIMENT_NAME}_launcher_${RUN_TS}.log"
TRAIN_OUT="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.out"
TRAIN_ERR="$LOG_DIR/${EXPERIMENT_NAME}_train_${RUN_TS}.err"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
ln -sfn "$(basename "$LAUNCHER_LOG")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.launcher.log"
ln -sfn "$(basename "$TRAIN_OUT")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.out"
ln -sfn "$(basename "$TRAIN_ERR")" "$LOG_DIR/${EXPERIMENT_NAME}.latest.train.err"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

cd "$RLLM_ROOT"

echo "==== CUDA preflight ===="
"$PYTHON_BIN" - "$NUM_GPUS" <<'PY'
import sys
import ctypes

import torch

required_gpus = int(sys.argv[1])
nccl = ctypes.CDLL("libnccl.so.2")
nccl_version = ctypes.c_int()
if nccl.ncclGetVersion(ctypes.byref(nccl_version)) != 0:
    raise SystemExit("Failed to query the loaded NCCL library version.")
torch_nccl = torch.cuda.nccl.version()
expected_nccl = (
    torch_nccl[0] * 10_000 + torch_nccl[1] * 100 + torch_nccl[2]
)
print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_nccl={torch_nccl}")
print(f"loaded_nccl={nccl_version.value}")
print(f"loaded_nccl_path={nccl._name}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_gpu_count={torch.cuda.device_count()}")

if nccl_version.value != expected_nccl:
    raise SystemExit(
        f"NCCL mismatch: PyTorch expects {expected_nccl}, but "
        f"libnccl.so.2 reports {nccl_version.value}. Reinstall the "
        "nvidia-nccl-cu12 package required by this PyTorch build."
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Check that the task node NVIDIA driver supports "
        f"the PyTorch CUDA runtime ({torch.version.cuda})."
    )
if torch.cuda.device_count() < required_gpus:
    raise SystemExit(
        f"Expected {required_gpus} visible GPUs, but found "
        f"{torch.cuda.device_count()}."
    )
PY

echo "==== DeepScaler Swift colocate test ===="
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "num_gpus=$NUM_GPUS"
echo "python_bin=$PYTHON_BIN"
echo "accelerate_bin=$ACCELERATE_BIN"
echo "model_path=$MODEL_PATH"
echo "project_name=$PROJECT_NAME"
echo "experiment_name=$EXPERIMENT_NAME"
echo "launcher_log=$LAUNCHER_LOG"
echo "train_out=$TRAIN_OUT"
echo "train_err=$TRAIN_ERR"
echo "checkpoint_dir=$CHECKPOINT_DIR"

# The environment-provided byted-wandb package replaces the standard wandb
# module, so the regular "wandb" tracking backend is used here.
"$ACCELERATE_BIN" launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m examples.deepscaler.train_deepscaler_swift \
    model.name="$MODEL_PATH" \
    model.trust_remote_code=true \
    rollout.mode=colocate \
    rollout.batch_size=16 \
    rollout.tensor_parallel_size=1 \
    rollout.gpu_memory_utilization=0.4 \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    training.group_size=8 \
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
