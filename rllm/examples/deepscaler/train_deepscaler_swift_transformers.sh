#!/usr/bin/env bash
# Single-machine Swift backend without external vLLM — batched transformers generate.
# Faster than the old TRL serial lock, but still slower than vLLM server mode.

set -x

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

python3 -m examples.deepscaler.train_deepscaler_swift \
    model.name=$MODEL_PATH \
    rollout.mode=transformers \
    rollout.batch_size=8 \
    data.train_batch_size=2 \
    training.group_size=4 \
    data.max_response_length=4096 \
    agent.max_steps=1 \
    trainer.logger=['console'] \
    trainer.experiment_name='deepscaler-swift-transformers' \
    trainer.default_local_dir=/tmp/rllm-swift-transformers
