#!/usr/bin/env bash
# DeepScaler with Swift backend — vLLM server (ms-swift style) + Accelerate train.
#
# Terminal A (inference GPUs, e.g. GPU 0,1):
#   CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --host 0.0.0.0 --port 8000 --dtype bfloat16 \
#     --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
#     --max-model-len 18432
#
# Terminal B (train GPUs, e.g. GPU 2,3):
#   CUDA_VISIBLE_DEVICES=2,3 bash examples/deepscaler/train_deepscaler_swift.sh

set -x

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
NUM_GPUS=${NUM_GPUS:-2}
VLLM_URL=${VLLM_URL:-http://127.0.0.1:8000/v1}

accelerate launch \
    --num_processes ${NUM_GPUS} \
    --multi_gpu \
    --mixed_precision bf16 \
    -m examples.deepscaler.train_deepscaler_swift \
    model.name=$MODEL_PATH \
    rollout.mode=server \
    rollout.base_url=$VLLM_URL \
    rollout.batch_size=16 \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    training.group_size=8 \
    training.learning_rate=1e-6 \
    sampling.temperature=0.6 \
    sampling.top_p=0.95 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    agent.max_steps=1 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-swift' \
    trainer.experiment_name='deepscaler-1.5b-swift' \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir=/tmp/rllm-swift-deepscaler
