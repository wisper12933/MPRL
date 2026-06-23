#!/bin/bash
#SBATCH --job-name=webshop_test
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -o ./logs/test/test_webshop_infer.out
#SBATCH -e ./logs/test/test_webshop_infer.err

module load cuda/12.9 
module load anaconda3/2023.3
module load jdk/11
source activate rllm

export PYTHONPATH=$(pwd)/..

LORA_NAME="webshop-lora-adapter"
LORA_PATH="/mnt/home/user28/llms/Qwen3-4B-Instruct-checkpoints/sft_web"

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/home/user28/llms/Qwen3-4B-Instruct/ \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-lora \
    --lora-modules $LORA_NAME=$LORA_PATH &

VLLM_PID=$!  

echo "等待 vLLM 服务器启动..."
MAX_RETRIES=120
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    RESPONSE=$(curl -s http://localhost:30000/v1/models 2>/dev/null)
    if echo "$RESPONSE" | grep -q "Qwen3-4B"; then
        echo "vLLM 服务器已就绪！"
        break
    fi
    
    if !  kill -0 $VLLM_PID 2>/dev/null; then
        echo "错误：vLLM 服务器启动失败"
        exit 1
    fi
    
    echo "等待中...  ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "错误：等待超时"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

python -m mprl.run_interact
SCRIPT_EXIT_CODE=$?  

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

exit $SCRIPT_EXIT_CODE