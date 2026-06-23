import asyncio

from transformers import AutoTokenizer
from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.agents.interact_agent import InteractAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.interact.webshop_env import WebShopEnv
from rllm.utils import compute_pass_at_k


""""
### ALFWorld
"train": "/mnt/home/user28/MPRL/data/indices/alfworld/train.json",
"test": "/mnt/home/user28/MPRL/data/indices/alfworld/test.json",
### SciWorld
"train": "/mnt/home/user28/MPRL/data/indices/sciworld/train.json",
"test": "/mnt/home/user28/MPRL/data/indices/sciworld/test.json",
### WebShop
"train": "/mnt/home/user28/MPRL/data/indices/webshop/train.json",
"test": "/mnt/home/user28/MPRL/data/indices/webshop/test.json",
"""
def prepare_data():
    data_files = {
        "train": "/mnt/home/user28/MPRL/data/indices/webshop/train.json",
        "test": "/mnt/home/user28/MPRL/data/indices/webshop/test.json",
    }
    datasets = load_dataset("json", data_files=data_files)

    train_dataset = DatasetRegistry.register_dataset("webshop", datasets["train"], "train")
    test_dataset = DatasetRegistry.register_dataset("webshop", datasets["test"], "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 4
    
    lora_real_path = "/mnt/home/user28/llms/Qwen3-4B-Instruct-checkpoints/sft_web"
    lora_model_name = "webshop-lora-adapter" 

    tokenizer = AutoTokenizer.from_pretrained(lora_real_path, trust_remote_code=True)

    env_args = {}

    sampling_params = {"temperature": 0.9, "top_p": 0.9, "model": lora_model_name}

    engine = AgentExecutionEngine(
        agent_class=InteractAgent,
        env_class=WebShopEnv,
        agent_args={"base_prompt_path": "/mnt/home/user28/MPRL/data/instructions/webshop_inst.txt"},
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
            "model": lora_model_name # it's necessary to set lora adapter name!!!
        },
        max_response_length=8192,
        max_prompt_length=8192,
        max_steps=12,
        n_parallel_agents=n_parallel_agents,
    )

    test_dataset = DatasetRegistry.load_dataset("webshop", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        _, test_dataset = prepare_data()

    tasks = test_dataset.repeat(n=4)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
