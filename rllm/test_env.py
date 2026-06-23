# test_vllm_full.py
import torch
from vllm import LLM, SamplingParams
import time

print("开始 vLLM 兼容性测试...")

# 测试参数
model_path = "/mnt/home/user28/llms/Qwen3-4B-Instruct"  # 例如: "meta-llama/Llama-2-7b-chat-hf" 或本地路径

try:
    # 1. 检查基本环境
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU可用: {torch.cuda.is_available()}")
    
    # 2. 初始化模型（使用小参数以快速测试）
    print(f"\n正在加载模型: {model_path}")
    start_time = time.time()
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # 单GPU
        gpu_memory_utilization=0.4,  # 使用较少内存
        max_num_seqs=2,
        max_model_len=512
    )
    
    load_time = time.time() - start_time
    print(f"模型加载成功! 耗时: {load_time:.2f}秒")
    
    # 3. 测试推理
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=50
    )
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is machine learning?"
    ]
    
    print("\n开始推理测试...")
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated: {output.outputs[0].text}")
    
    print("\n✅ vLLM 兼容性测试完全通过!")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("可能是版本不兼容，请检查 vLLM 和 PyTorch 版本")
except RuntimeError as e:
    print(f"❌ 运行时错误: {e}")
    print("可能是 CUDA 版本不匹配或内存不足")
except Exception as e:
    print(f"❌ 其他错误: {e}")