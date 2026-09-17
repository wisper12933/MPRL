# Swift Backend（ms-swift 风格，无 Ray）

解决 TRL backend 的两大痛点：**串行采样**、**采样吞吐低**。  
训练仍用 Accelerate 多卡；采样对齐 ms-swift：优先 **vLLM server 并发**，其次 colocate / batched transformers。

参考：[ms-swift GRPO](https://github.com/modelscope/ms-swift)

## 和 `trl` 的对比

| 问题 | `trl` backend | `swift` backend |
|------|---------------|-----------------|
| GPU generate | `_gen_lock` 串行，`batch=1` | **server 并发** / **请求合批** |
| logprobs | generate 后再 forward | generate/scores 一次拿回 |
| 多卡训练 | Accelerate DDP | 同样 Accelerate DDP |
| 推荐部署 | 单进程 HF | **推理卡跑 vLLM，训练卡跑 Accelerate**（像 ms-swift server） |

**agent / env / reward 无需修改**，仍走 `AsyncAgentExecutionEngine`。

---

## 目录

```
rllm/engine/rollout/swift_engine.py   # server | colocate | transformers
rllm/trainer/swift/
  swift_agent_trainer.py
  swift_policy_trainer.py
  swift_data_processor.py
rllm/trainer/config/swift_rl_trainer.yaml
examples/deepscaler/train_deepscaler_swift.py
examples/deepscaler/train_deepscaler_swift.sh
```

---

## Rollout 三种模式

### 1. `server`（推荐，对齐 ms-swift `--vllm_mode server`）

vLLM OpenAI HTTP，**多个 async 请求并行**，无全局锁。

```bash
# 推理卡
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --port 8000 --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 18432

# 训练卡
CUDA_VISIBLE_DEVICES=2,3 bash examples/deepscaler/train_deepscaler_swift.sh
```

配置：`rollout.mode=server rollout.base_url=http://127.0.0.1:8000/v1`

> 权重同步：每步后 train checkpoint 与 vLLM 进程独立。生产上可像 ms-swift 一样另做 NCCL 同步；调试/短跑可定期 reload vLLM，或先用 `transformers` 模式保证权重一致。

### 2. `colocate`（对齐 ms-swift colocate）

进程内 `vllm.LLM`，请求 **合批 generate**。

```bash
rollout.mode=colocate rollout.batch_size=16
# 需要: pip install vllm
```

### 3. `transformers`（无 vLLM 时的提速 fallback）

去掉 TRL 的串行锁，用 **micro-batch + `output_scores`**（不再二次 forward）。

```bash
bash examples/deepscaler/train_deepscaler_swift_transformers.sh
```

仍慢于 vLLM，但明显快于旧 TRL 串行路径。

---

## 安装

```bash
pip install -e ".[swift]"
# server/colocate 另装
pip install vllm
```

可选：完整 ms-swift CLI `pip install ms_swift`（见文末 bridge）。

---

## DeepScaler 快速开始

```bash
python examples/deepscaler/prepare_math_data.py
bash examples/deepscaler/train_deepscaler_swift.sh
```

代码入口：

```python
AgentTrainer(..., backend="swift", config=config)  # config_name=swift_rl_trainer
```

---

## 关键配置

| 键 | 含义 |
|----|------|
| `rollout.mode` | `server` / `colocate` / `transformers` |
| `rollout.base_url` | vLLM OpenAI URL |
| `rollout.batch_size` | colocate/transformers 合批大小 |
| `training.group_size` | GRPO 每 prompt 采样数（原 `rollout.n`） |
| `data.train_batch_size` | **每 GPU** prompt 数 |
| `agent.max_steps` | agent↔env 交互轮数（DeepScaler=1） |

多卡：

```bash
accelerate launch --num_processes 4 --multi_gpu -m examples.deepscaler.train_deepscaler_swift ...
```

---

## 数据流

```
AgentTrainer(backend="swift")
  └─ SwiftAgentTrainer
       ├─ AsyncAgentExecutionEngine(engine_name="swift")  ← 原 agent/env
       │    └─ SwiftEngine  ← 并发/合批采样 + logprobs
       └─ SwiftPolicyTrainer (Accelerate DDP)  ← GRPO update
```

---

## 与直接使用 `ms_swift` CLI 的关系

本 backend **复用 rLLM 的 AgentExecutionEngine**，采样/训练组织借鉴 ms-swift，但**不强制**跑 `swift rlhf`。

若你希望 100% 走 ms-swift 的 `GRPOTrainer` + `gym_scheduler`：

1. `pip install ms_swift`
2. 把 rLLM env 包一层成 `swift.rollout.gym_env.Env`（参考 ms-swift `FrozenLakeEnv`）
3. `--external_plugins your_plugin.py --multi_turn_scheduler gym_scheduler --use_vllm true`

本仓库先提供 **嵌在 AgentTrainer 里、可改 agent/env、无 Ray** 的路径；需要完整 ms-swift bridge 时可再加。

---

## 故障排查

| 现象 | 处理 |
|------|------|
| 连不上 vLLM | 检查 `rollout.base_url`、端口、`CUDA_VISIBLE_DEVICES` 是否拆分 |
| OOM（transformers） | 减小 `batch_size` / `group_size` / `max_response_length`，或改 `server` |
| 采样仍慢 | 确认不是 `transformers`；应看 GPU 上有多个 concurrent vLLM 请求 |
| import SwiftEngine | `pip install -e ".[swift]"` |
