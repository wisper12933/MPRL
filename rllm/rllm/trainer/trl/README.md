# TRL Backend（无 Ray）

本文档说明 rLLM 新增的 `backend="trl"` 训练路径，供开发机上的 agent 或开发者快速上手。

## 背景与动机

公司开发机上使用默认 `verl` backend 时，**Ray 集群通信受限**（端口/GCS/进程间通信），难以调试且无法稳定训练。

`trl` backend 的设计目标：

- **完全不依赖 Ray**
- **复用现有 `AsyncAgentExecutionEngine`**，agent class / env class / reward 逻辑**无需修改**
- 使用本地 HuggingFace 模型做 rollout + GRPO 风格 policy update
- API 与 `AgentTrainer` 保持一致，仅切换 `backend="trl"` 和配置文件

## 与 verl backend 对比

| 项目 | `verl`（默认） | `trl`（新增） |
|------|----------------|---------------|
| 分布式编排 | Ray + RayWorkerGroup | 单进程，无 Ray |
| Rollout 引擎 | vLLM hybrid engine | 本地 `model.generate()` |
| 训练引擎 | FSDP / Megatron (verl) | HuggingFace + AdamW（可选 LoRA） |
| 算法 | GRPO / PPO 等 | GRPO advantage + clipped policy gradient |
| Agent/Env | `AsyncAgentExecutionEngine` | **同一套** `AsyncAgentExecutionEngine` |
| 多卡 | 支持（8 GPU 等） | **Accelerate DDP**（`accelerate launch --num_processes=N`） |
| 长上下文 offload | 支持 | 受单卡显存限制 |

适合场景：开发机调试、Ray 不可用环境、中小规模单卡实验。  
不适合场景：需要 verl 全量 FSDP + vLLM hybrid + 多机多卡的 DeepScaler 生产训练。

---

## 目录结构（本次新增/修改）

```
rllm/
├── engine/
│   ├── agent_execution_engine.py   # [修改] 新增 engine_name="trl"
│   └── rollout/
│       └── trl_engine.py           # [新增] 本地 HF rollout
├── trainer/
│   ├── agent_trainer.py            # [修改] backend 增加 "trl"
│   ├── config/
│   │   └── trl_rl_trainer.yaml     # [新增] TRL 专用 Hydra 配置
│   └── trl/
│       ├── README.md               # 本文档
│       ├── __init__.py
│       ├── trl_agent_trainer.py    # 训练主循环（仿 tinker 结构）
│       ├── trl_policy_trainer.py   # 模型加载、checkpoint、梯度更新
│       └── trl_data_processor.py   # Episode → sample，GRPO advantage
├── examples/deepscaler/
│   ├── train_deepscaler_trl.py     # [新增] DeepScaler 入口
│   └── train_deepscaler_trl.sh     # [新增] 启动脚本
└── pyproject.toml                  # [修改] 增加 optional-dependencies.trl
```

---

## 架构数据流

```
AgentTrainer(backend="trl")
    │
    ▼
TrlAgentTrainer
    │
    ├─► AsyncAgentExecutionEngine (engine_name="trl")
    │       │
    │       ├─ agent_class / env_class  ← 用户自定义，与 verl 相同
    │       ├─ env.reset() / env.step() / reward_fn
    │       └─ TrlEngine.get_model_response()  ← 本地 HF generate + logprobs
    │
    └─► TrlPolicyTrainer.step()
            ├─ process_episodes()     ← GRPO grouping & advantage
            └─ importance_sampling_loss + optimizer.step()
```

**要点：** rollout 与 policy update 共用同一 HF 模型；每次 train step 后会 `set_model()` 同步权重到 rollout engine。

---

## 安装

在 rLLM 项目根目录（含 `pyproject.toml` 的目录）执行：

```bash
# 安装 rLLM + TRL backend 依赖
uv pip install -e ".[trl]"
# 或
pip install -e ".[trl]"
```

`[trl]` 额外依赖：`trl>=0.17.0`、`peft>=0.14.0`、`accelerate>=1.2.0`。

**不需要**安装 verl、vLLM、Ray（若只做 trl 训练）。

---

## 快速开始：DeepScaler 示例

### 1. 准备数据

```bash
cd examples/deepscaler
python prepare_math_data.py
```

会注册 `deepscaler_math`（train）和 `aime2024`（test）到 `DatasetRegistry`。

### 2. 启动训练

```bash
# 在 rllm 包根目录下
bash examples/deepscaler/train_deepscaler_trl.sh
```

或直接：

```bash
python -m examples.deepscaler.train_deepscaler_trl \
    model.name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    data.train_batch_size=4 \
    training.group_size=4 \
    data.max_response_length=16384
```

### 3. 自定义任务（任意 agent/env）

```python
from rllm.trainer import AgentTrainer

trainer = AgentTrainer(
    config=config,                    # Hydra: config_name="trl_rl_trainer"
    agent_class=YourAgent,            # 不变
    env_class=YourEnv,                # 不变
    agent_args={...},
    env_args={...},
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    backend="trl",                    # 关键：切换 backend
)
trainer.train()
```

Hydra 入口模板：

```python
@hydra.main(
    version_base=None,
    config_path="pkg://rllm.trainer.config",  # 或相对路径
    config_name="trl_rl_trainer",
)
def main(config):
    ...
```

---

## 配置说明（`trl_rl_trainer.yaml`）

### 模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.name` | DeepSeek-R1-Distill-Qwen-1.5B | HF 模型路径或 ID |
| `model.use_lora` | `false` | 显存不足时可设 `true` |
| `model.lora_rank` | 32 | LoRA rank |
| `model.gradient_checkpointing` | `true` | 节省显存 |

### 训练

| 参数 | 默认值 | 对应 verl 概念 |
|------|--------|----------------|
| `training.group_size` | 8 | `actor_rollout_ref.rollout.n` |
| `training.learning_rate` | 1e-6 | actor lr |
| `training.clip_ratio` | 0.2 | PPO clip |
| `training.num_minibatches` | 1 | 每个 batch 内梯度更新切分 |

### 采样（rollout）

| 参数 | 默认值 |
|------|--------|
| `sampling.temperature` | 0.6 |
| `sampling.top_p` | 0.95 |

### 算法

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | `grpo` | 支持 `grpo` / `reinforce` |
| `algorithm.norm_adv_by_std_in_grpo` | `false` | DeepScaler 论文设置 |
| `algorithm.grouping_level` | `episode` | advantage 分组粒度 |

### 数据与 agent

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.train_batch_size` | 16 | 每步 prompt 数 |
| `data.max_prompt_length` | 2048 | |
| `data.max_response_length` | 16384 | 可按 8K→16K→24K 迭代加长 |
| `agent.max_steps` | 1 | 单轮 math 为 1；多轮任务增大 |
| `agent.n_parallel_agents` | `null` | 默认 `train_batch_size * group_size` |

### Checkpoint

| 参数 | 默认值 |
|------|--------|
| `trainer.default_local_dir` | `/tmp/rllm-trl-checkpoints` |
| `trainer.save_freq` | 20 |
| `trainer.test_freq` | 20 |

Checkpoint 目录结构：

```
{default_local_dir}/
├── latest_batch.txt          # 记录最后保存的 batch id
└── checkpoint-{batch_idx}/   # HF save_pretrained + optimizer.pt
```

---

## Agent 开发机操作清单

按顺序执行，避免常见错误：

1. **确认 GPU 可用**：`nvidia-smi`
2. **安装依赖**：`pip install -e ".[trl]"`
3. **准备数据集**：对应 example 的 `prepare_*` 脚本
4. **从小 batch 试跑**：`data.train_batch_size=2 training.group_size=2`
5. **观察日志**：`reward/mean`、`loss/policy`、`time/sample`、`time/train`
6. **显存 OOM 时**：
   - 减小 `train_batch_size` / `group_size`
   - 减小 `max_response_length`
   - 开启 `model.use_lora=true`
   - 保持 `model.gradient_checkpointing=true`

---

## 多卡训练（Accelerate DDP）

TRL backend 已集成 **HuggingFace Accelerate**，用 DDP 同步训练梯度；每张卡独立做 rollout + 本地 backward。

### 启动方式

```bash
# 4 卡示例（推荐直接用脚本）
bash examples/deepscaler/train_deepscaler_trl_4gpu.sh

# 或手动指定
accelerate launch \
    --num_processes 4 \
    --multi_gpu \
    --mixed_precision bf16 \
    -m examples.deepscaler.train_deepscaler_trl \
    data.train_batch_size=4 \
    training.group_size=4 \
    ...
```

也可用 `torchrun`（Accelerate 会自动识别环境）：

```bash
torchrun --nproc_per_node=4 -m examples.deepscaler.train_deepscaler_trl ...
```

### Batch 语义（重要）

| 参数 | 含义 |
|------|------|
| `data.train_batch_size` | **每张 GPU** 上的 prompt 数 |
| `training.group_size` | 每个 prompt 的 rollout 条数（GRPO 分组） |
| 有效全局 prompt 数 | `train_batch_size × num_gpus` |
| 有效全局 rollout 数 | `train_batch_size × num_gpus × group_size` |

例：4 卡、`train_batch_size=4`、`group_size=4` → 每步 16 个 prompt、64 条 trajectory。

### 多卡行为说明

- 各 rank 通过 `DistributedSampler` 切分数据，**不会重复采样同一 prompt**
- 各 rank 在本卡 GPU 上用 `TrlEngine` 做 rollout
- `TrlPolicyTrainer` 用 `accelerator.prepare(model, optimizer)` 做 DDP
- **仅 rank 0** 写 checkpoint / wandb / console log
- `reward/mean` 等指标会跨卡 `reduce` 后记录

### 单卡 vs 多卡

| 场景 | 启动命令 |
|------|----------|
| 单卡 | `python -m examples.deepscaler.train_deepscaler_trl ...` |
| 多卡 | `accelerate launch --num_processes 4 ...` |

单卡时 Accelerate 自动退化为 `num_processes=1`，无需改代码。

---

## 修改现有 example 的检查项

若 agent 需要把其他 example 从 verl 迁到 trl：

- [ ] `backend="trl"`
- [ ] Hydra `config_name` 改为 `trl_rl_trainer`（不要用 `agent_ppo_trainer`）
- [ ] verl 专有参数（`actor_rollout_ref.*`、`ray_init.*`、`trainer.n_gpus_per_node`）**无效**，勿传入
- [ ] `training.group_size` 替代 `actor_rollout_ref.rollout.n`
- [ ] agent / env / reward 代码**不用改**

---

## 已知限制与后续计划

**当前限制：**

- 多卡为 **DDP 数据并行**，无 FSDP / DeepSpeed 分片
- Rollout 用各卡本地 `model.generate()`，长序列比 vLLM 慢
- 无 verl 的 rejection sampling、stepwise advantage 等高级配置
- 每卡 rollout 与 train 串行共享该卡 GPU

**可扩展方向：**

- 外接 vLLM server 做 rollout（训练仍用 HF）
- FSDP / DeepSpeed 支持更大模型
- 更完整对齐 verl 的 metric / wandb 字段

---

## 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `Unsupported backend` | 未更新 `agent_trainer.py` 或旧安装 | 重新 `pip install -e .` |
| `Datasets not found` | 未跑 prepare 脚本 | 运行 `prepare_math_data.py` |
| CUDA OOM | batch 或 response 过长 | 减小 batch / group_size / max_response_length，或开 LoRA |
| `engine_name ... not supported` | `agent_execution_engine.py` 未更新 | 确认含 `"trl"` 分支 |
| 训练无梯度 / loss=0 | 组内 reward 全相同 | 正常；可调 `algorithm.remove_constant_reward_groups` |
| import `peft` 失败 | 未装 `[trl]` extra | `pip install -e ".[trl]"` |

---

## 相关文件速查

| 用途 | 路径 |
|------|------|
| 用户入口 API | `rllm/trainer/agent_trainer.py` → `_train_trl()` |
| 训练循环 | `rllm/trainer/trl/trl_agent_trainer.py` |
| 策略更新 | `rllm/trainer/trl/trl_policy_trainer.py` |
| Rollout | `rllm/engine/rollout/trl_engine.py` |
| 执行引擎 | `rllm/engine/agent_execution_engine.py` |
| 默认配置 | `rllm/trainer/config/trl_rl_trainer.yaml` |
| DeepScaler 示例 | `examples/deepscaler/train_deepscaler_trl.py` |

---

## 参考：verl vs trl 启动命令对照

**verl（原 DeepScaler，需要 Ray）：**

```bash
python -m examples.deepscaler.train_deepscaler \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=8 \
  ...
```

**trl（无 Ray）：**

```bash
python -m examples.deepscaler.train_deepscaler_trl \
    algorithm.adv_estimator=grpo \
    training.group_size=8 \
  ...
```
