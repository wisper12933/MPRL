set -x

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

python3 -m examples.deepscaler.train_deepscaler_trl \
    model.name=$MODEL_PATH \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    training.group_size=4 \
    training.learning_rate=1e-6 \
    training.num_minibatches=1 \
    sampling.temperature=0.6 \
    sampling.top_p=0.95 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    agent.max_steps=1 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-trl' \
    trainer.experiment_name='deepscaler-1.5b-16k-trl' \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir=/tmp/rllm-trl-deepscaler
