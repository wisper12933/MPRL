# MPRL：Meta-Planning Reinforcement Learning

本文档说明本仓库围绕 Qwen3-4B-Instruct 实现的三阶段训练与评测流程：

1. **Meta-Plan MAML**：在 ALFWorld、WebShop 和 ScienceWorld 的规划数据上学习跨任务步骤级规划能力。
2. **下游任务 SFT**：分别在三个 benchmark 的交互轨迹上微调，得到互相独立的任务 adapter。
3. **交互式 RL**：使用 rLLM、Swift backend 和外部 vLLM rollout server，对每个任务的 adapter 分别执行 GRPO 训练。

第三阶段支持可选的 Meta-Plan。默认开启：环境给出初始任务后，模型先生成一次 `<workflow>`，将其注入初始上下文，再进行正常的 Thought/Action 交互。规划生成本身不会形成 trainable trajectory step，因此不参与 policy-gradient 更新。

---

## 1. 项目结构

从仓库根目录看，主要文件如下：

```text
MPRL/
├── data/
│   ├── raw/                         # 原始 Meta-Plan 数据
│   ├── step1_metaplan/
│   │   ├── train/                   # MAML 训练/验证数据
│   │   └── test/                    # Meta-Plan 独立测试数据
│   ├── step2_sft/                   # 三个任务各自的交互 SFT 数据
│   ├── indices/                     # RL/benchmark 的 train/test 实例划分
│   ├── instructions/                # 交互 prompt
│   │   └── metaplan/                # RL 阶段的 Meta-Plan prompt 模板
│   └── alfworld_data/               # ALFWorld 环境数据
├── envs/
│   ├── alfworld/
│   └── webshop/
├── maml/                            # 第一阶段 MAML 和 benchmark 评测代码
├── scripts/                         # MAML/benchmark 启动脚本
└── rllm/
    ├── mprl/                        # 第三阶段 MPRL 入口和任务定义
    ├── rllm/                        # rLLM 核心、Swift trainer 和环境封装
    ├── run_mprl_swift_rollout_server.sh
    ├── run_train_mprl_swift_server.sh
    └── run_infer_demo.sh
```

第三阶段的关键代码：

- `task_specs.py`：任务、adapter、数据划分、prompt、环境类和默认步数的统一映射。
- `planned_interact_agent.py`：一次性 Meta-Plan 生成、规范化和上下文注入。
- `train_interact.py`：Hydra 训练入口。
- `run_interact.py`：不训练、只进行交互采样的入口。
- `prepare_data.py`：检查和规范化任务索引。
- `../rllm/trainer/config/mprl_swift_trainer.yaml`：第三阶段默认训练配置。

---

## 2. 模型与 adapter

默认基础模型：

```text
/mnt/hdfs/.../qwen_model/origin/Qwen3-4B-Instruct
```

默认 adapter 根目录：

```text
/mnt/hdfs/.../qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora
```

三个任务必须独立训练，并使用各自的 adapter：

- WebShop：`Qwen3-4B-Instruct-MAML-plan-sft-web`
- ALFWorld：`Qwen3-4B-Instruct-MAML-plan-sft-alf`
- ScienceWorld：`Qwen3-4B-Instruct-MAML-plan-sft-sci`

启动脚本会根据 `TASK=webshop|alfworld|sciworld` 自动选择上述目录。也可以通过 `BASE_MODEL`、`ADAPTER_ROOT` 或 `ADAPTER_PATH` 覆盖。

> 不要使用一个任务的 adapter 训练另一个任务。三个任务的动作空间、prompt、奖励和数据划分不同，第三阶段的 checkpoint 也应分别保存。

---

## 3. 环境准备

### 3.1 Python 环境

项目环境固定为 Python 3.11，当前约定的虚拟环境目录为：

```text
.venv-mprl311
```

从仓库根目录创建环境：

```bash
uv venv --python 3.11 .venv-mprl311
uv pip install --python .venv-mprl311/bin/python -r requirements.txt
uv pip install --python .venv-mprl311/bin/python -e './rllm[swift]'
```

`swift` extra 会安装本项目权重同步代码所依赖的 `ms-swift==3.12.3`。只执行 `-e ./rllm` 不足以保证 `swift rollout` 命令存在。

当前已验证的关键版本包括：

- Python 3.11
- PyTorch 2.8.0 + CUDA 12.8
- Transformers 4.57.3
- Accelerate 1.12.0
- vLLM 0.11.0
- PEFT 0.18.1
- ms-swift/Swift rollout
- NumPy 1.26.4
- Pyserini 0.17.0
- ScienceWorld 1.1.3

安装后至少执行：

```bash
.venv-mprl311/bin/python - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
PY
```

### 3.2 Byted-WandB

公司环境使用 `byted-wandb` 替代公开版 WandB。训练脚本已经设置：

```bash
WANDB_DISABLE_SERVICE=true
WANDB_START_METHOD=thread
```

这样可以绕过 byted-wandb service subprocess 的临时端口文件问题，同时仍上传到公司实验平台。

确认环境：

```bash
uv pip show --python .venv-mprl311/bin/python byted-wandb
```

如果使用 byted-wandb，不要在训练前再次用公开 PyPI 的 `wandb` 覆盖它。日志中的 HTTPS、DNS 或 SSL warning 通常表示公司网络链路暂时不可用；本地训练和本地日志通常仍可继续。

### 3.3 Java 11

WebShop 的 Pyserini/PyJNIus 和 ScienceWorld 都依赖 Java。默认路径为：

```text
/opt/tiger/jdk/jdk11
```

检查：

```bash
/opt/tiger/jdk/jdk11/bin/java -version
test -f /opt/tiger/jdk/jdk11/lib/server/libjvm.so
```

训练脚本默认固定使用 Java 11，避免继承旧的 Java 8 环境。自定义路径时使用：

```bash
export MPRL_JAVA_HOME=/path/to/jdk11
export MPRL_JVM_PATH=/path/to/jdk11/lib/server/libjvm.so
```

### 3.4 GPU 分配

推荐将 rollout 和训练放在不同 GPU：

- GPU 0：Swift/vLLM rollout server
- GPU 1、2、3：Accelerate DDP 训练

对应默认参数：

```bash
VLLM_GPUS=0
TRAIN_GPUS=1,2,3
```

不要让两组 GPU 重叠，否则 vLLM KV cache 与训练模型、梯度和 optimizer state 会竞争显存。

### 3.5 环境资产检查

ALFWorld：

```bash
test -d data/alfworld_data
test -f envs/alfworld/base_config.yaml
```

ScienceWorld：

```bash
test -f data/indices/sciworld/scienceworld.jar
```

WebShop 需要 `envs/webshop/web_agent_site/data/` 下的商品、属性和搜索索引文件，主要包括：

```text
items_shuffle.json
items_ins_v2.json
items_human_ins.json
```

---

## 4. Benchmark 说明

三个 benchmark 的独立评测脚本都会报告环境 reward；`mprl.run_interact` 还会将每个测试任务重复 `REPEAT_K` 次并调用 `compute_pass_at_k()`。因此：

- reward mean 衡量平均完成质量；
- success rate 衡量完全成功的比例；
- pass@k 衡量同一任务采样 k 次时至少成功一次的能力。

比较不同实验时必须保持测试索引、重复次数和采样参数一致。增加 k 通常会提高 pass@k，不能与较小 k 的结果直接横向比较。

### 4.1 WebShop

WebShop 是文本网页购物环境。模型根据用户约束搜索商品、进入详情页、选择属性并购买。

- 动作：`search[keywords]`、`click[value]`
- 训练实例：1,824
- 测试实例：200
- 默认最大交互步数：12
- 默认并行 agent 数：2
- 主要指标：
  - 平均 reward：商品匹配程度的连续分数
  - success rate：reward 等于 1.0 的比例

索引格式是整数商品任务 ID，例如：

```json
2853
```

WebShop 数据加载成本较高。rLLM 环境封装在同一个训练进程内共享 `SimServer`，并使用锁保护并发请求；每个 DDP rank 仍会独立加载一份 WebShop 数据。

### 4.2 ALFWorld

ALFWorld 是文本化家庭环境任务，包括拾取、放置、清洁、加热、冷却和多物体操作等类型。

- 训练实例：3,119
- 测试实例：134
- 训练 split：`train`
- 测试 split：`eval_out_of_distribution`
- 默认最大交互步数：40
- 默认并行 agent 数：2
- 主要指标：`won`，成功为 1，失败为 0

索引中保存 `game_file`。加载时，代码会截取旧路径中的 `/alfworld_data/` 后缀，并自动映射到当前 `data/alfworld_data`。

RL 不需要 ALFWorld 的 hand-coded DAgger expert。`envs/alfworld/base_config.yaml` 中应保持：

```yaml
env:
  expert_plan: False
```

### 4.3 ScienceWorld

ScienceWorld 是包含物理、化学、电学和生物实验的文本科学环境。

- 训练实例：1,483
- 测试实例：211
- 默认最大交互步数：60
- 默认并行 agent 数：1
- 索引格式：`[task_name, variation_idx]`
- 主要指标：
  - 平均归一化任务分数
  - score 为 1.0 的成功率

示例：

```json
["task-1-boil", 21]
```

RL 环境内部将 ScienceWorld 的累计 score 转换为逐步 delta reward，避免在每一步重复累计历史得分。

---

## 5. 数据说明

### 5.1 第一阶段 Meta-Plan 数据

目录：

```text
data/step1_metaplan/train/
```

数据量：

- `alfworld_metaplan_train.json`：2,507
- `webshop_metaplan_train.json`：1,523
- `sciworld_metaplan_train.json`：1,183
- `metaplan_eval.json`：600

独立测试集：

```text
data/step1_metaplan/test/metaplan_test.json
```

数据采用 ShareGPT 风格：

```json
{
  "id": "example-id",
  "conversations": [
    {
      "from": "human",
      "value": "Please generate a step-by-step workflow ... <task>...</task>"
    },
    {
      "from": "gpt",
      "value": "<workflow>\nStep 1: ...\n</workflow>"
    }
  ]
}
```

三个 benchmark 使用不同的任务前缀：

- WebShop：`Please generate a step-by-step workflow for a web shopping task:`
- ALFWorld：`Please generate a step-by-step workflow for a house holding task:`
- ScienceWorld：`Please generate a step-by-step workflow for a scientific task:`

### 5.2 第二阶段下游 SFT 数据

目录：

```text
data/step2_sft/
```

数据量与 RL 训练索引对齐：

- `alfworld_sft.json`：3,119
- `webshop_sft.json`：1,824
- `sciworld_sft.json`：1,483

数据包含完整多轮交互，模型回答采用：

```text
Thought: ...
Action: ...
```

仓库当前保存了第二阶段数据和最终 adapter 路径约定，但没有提供一套统一的第二阶段 SFT launcher。使用外部 SFT 工具时，必须分别训练三个 adapter，并保持本 README 第 2 节中的目录名，或在 RL 启动时显式传入 `ADAPTER_PATH`。

### 5.3 第三阶段 RL 索引

目录：

```text
data/indices/{alfworld,webshop,sciworld}/
```

每个任务都包含：

```text
train_indices.json
test_indices.json
```

`TRAIN_LIMIT=0` 表示不截断训练集；`VAL_LIMIT=16` 表示默认只使用测试集前 16 个实例进行周期验证。限制值大于 0 时取索引文件的前 N 项。

可以在正式训练前检查索引解析和 ALFWorld 路径重写：

```bash
cd /path/to/MPRL/rllm
PYTHONPATH="$PWD:.." ../.venv-mprl311/bin/python \
  -m mprl.prepare_data \
  --task alfworld \
  --alfworld-data ../data/alfworld_data \
  --limit 2
```

### 5.4 RL prompt

正常交互 prompt：

```text
data/instructions/alfworld_inst.txt
data/instructions/webshop_inst.txt
data/instructions/sciworld_inst.txt
```

Meta-Plan 模板：

```text
data/instructions/metaplan/alfworld.txt
data/instructions/metaplan/webshop.txt
data/instructions/metaplan/sciworld.txt
```

每个 Meta-Plan 模板必须包含 `{{TASK}}`。运行时用环境初始 observation 替换该占位符。

---

## 6. 第一阶段：Meta-Plan MAML

### 6.1 算法流程

`maml/trainer.py` 实现二阶 MAML：

1. 每个 batch 分成 support set 和 query set。
2. 在 support set 上执行 inner-loop 更新。
3. 使用更新后的 functional parameters 在 query set 上计算 meta loss。
4. `create_graph=True` 保留 inner update 的高阶梯度。
5. 对 meta loss 反向传播，更新原始可训练参数。

默认 MAML 参数：

- `support_size=2`
- `query_size=2`
- `inner_steps=1`
- `inner_lr=1e-4`

因此 `per_device_train_batch_size` 必须等于：

```text
support_size + query_size = 4
```

当前 `maml/configs/training_config.yaml` 使用三个任务数据，并通过 `interleave_over` 交错采样，使较短数据集可以重复采样到最长数据集结束。

### 6.2 修改配置

运行前检查：

```text
maml/configs/training_config.yaml
```

至少修改：

```yaml
model_name_or_path: /mnt/hdfs/.../qwen_model/origin/Qwen3-4B-Instruct
output_dir: /path/to/maml-output
```

数据配置应保持：

```yaml
dataset: alfworld_metaplan_train.json, sciworld_metaplan_train.json, webshop_metaplan_train.json
eval_dataset: metaplan_eval.json
dataset_dir: data/step1_metaplan/train
template: qwen3
per_device_train_batch_size: 4
```

`dataset_dir` 在参数类中的默认值是 `data/step1_metaplan/train`；建议在 YAML 中显式写出，避免从其他工作目录启动时产生歧义。

### 6.3 启动 MAML

从仓库根目录运行：

```bash
PYTHONPATH="$PWD" .venv-mprl311/bin/python \
  -m maml.run_training \
  --config ./maml/configs/training_config.yaml
```

旧的 `scripts/run_training.sh` 包含特定集群的 Slurm、module 和绝对路径配置，只适用于对应旧环境。新机器上优先使用上面的直接命令，或先修改该脚本。

### 6.4 Meta-Plan 评测

先修改：

```text
maml/configs/metaplan_eval_config.yaml
```

确保基础模型和 adapter 指向本次产物，然后运行：

```bash
PYTHONPATH="$PWD" .venv-mprl311/bin/python \
  -m maml.run_metaplan_eval \
  --config ./maml/configs/metaplan_eval_config.yaml
```

Meta-Plan 文本评测输出：

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU-4

这些指标衡量生成 workflow 与参考 workflow 的文本重合度，不等价于环境成功率。最终能力仍应通过三个交互 benchmark 评测。

---

## 7. 第二阶段：任务独立 SFT

第二阶段从同一个 Meta-Plan MAML 模型出发，分别使用：

```text
data/step2_sft/alfworld_sft.json
data/step2_sft/webshop_sft.json
data/step2_sft/sciworld_sft.json
```

训练后应形成三个独立 LoRA adapter：

```text
Qwen3-4B-Instruct-MAML-plan-sft-alf
Qwen3-4B-Instruct-MAML-plan-sft-web
Qwen3-4B-Instruct-MAML-plan-sft-sci
```

这一阶段的目标是让已经具有跨任务规划能力的模型适应各环境的：

- system/interaction prompt
- 动作语法
- observation 格式
- 多轮交互轨迹

第三阶段 RL 会把对应 SFT adapter 以 `is_trainable=True` 加载，因此 RL 是在该任务 adapter 上继续优化，不会重新创建一个随机 LoRA。

---

## 8. 第三阶段：带可选规划的交互式 RL

### 8.1 一条 trajectory 的执行顺序

启用 Plan 时：

```text
env.reset()
  -> 得到初始任务 observation
  -> 使用任务专属 Meta-Plan prompt 生成 <workflow>
  -> 将 workflow 注入初始 user context
  -> 正常生成 Thought/Action
  -> env.step(action)
  -> 重复交互直到完成或达到 max_steps
  -> 根据环境 reward 计算 GRPO advantage
  -> 只对正常交互 response token 做 policy-gradient 更新
```

禁用 Plan 时：

```text
env.reset()
  -> 得到初始任务 observation
  -> 直接开始 Thought/Action 交互
  -> 后续 RL 流程不变
```

### 8.2 为什么规划不参与 policy gradient

规划请求由 `AgentExecutionEngine._run_initial_planning()` 单独执行。生成结果只写入：

- agent 的初始上下文
- trajectory 的 `info["metaplan"]`

它不会被添加到 `trajectory.steps`。Swift data processor 只从 `trajectory.steps` 构造训练样本和 response mask，因此规划 token 不会进入 RL loss。

注意：后续动作生成会读取注入后的 workflow，所以规划会间接改变采样轨迹和最终 reward；“不参与梯度”并不表示它对策略行为没有影响。

### 8.3 GRPO 与权重同步

默认配置：

```yaml
algorithm:
  adv_estimator: grpo
  grouping_level: episode

training:
  group_size: 8
  num_minibatches: 8
  learning_rate: 1.0e-6
```

训练使用 Accelerate DDP。不同 rank 的环境轨迹长度可能不同，因此 Swift trainer 会保证所有 rank 执行相同数量的 backward collective，避免 DDP collective 次序错位。

外部 rollout server 使用官方 Swift communicator 接收 adapter 权重：

1. 训练前同步当前 adapter。
2. 完成 optimizer update 后递增 `policy_version`。
3. 新 `policy_version` 触发下一次权重传输。
4. vLLM 重置 prefix cache，后续 rollout 使用更新后的策略。

默认 `weight_sync_mode=auto`。当前模型是 PEFT/LoRA 模型，因此自动选择 adapter-only 同步。rollout server 必须以 `--vllm_enable_lora true` 启动。

---

## 9. 第三阶段启动方式

所有以下命令都从 `rllm/` 目录执行：

```bash
cd /path/to/MPRL/rllm
```

### 9.1 Terminal 1：启动 Swift rollout server

WebShop：

```bash
TASK=webshop VLLM_GPUS=0 ./run_mprl_swift_rollout_server.sh
```

ALFWorld：

```bash
TASK=alfworld VLLM_GPUS=0 ./run_mprl_swift_rollout_server.sh
```

ScienceWorld：

```bash
TASK=sciworld VLLM_GPUS=0 ./run_mprl_swift_rollout_server.sh
```

等待日志出现：

```text
Application startup complete
```

然后检查：

```bash
curl -f http://127.0.0.1:8000/health/
curl -f http://127.0.0.1:8000/get_world_size/
```

默认 server 参数：

- `VLLM_GPUS=0`
- `VLLM_PORT=8000`
- `VLLM_GPU_MEMORY_UTILIZATION=0.85`
- `MAX_MODEL_LEN=10240`
- tensor parallel size 自动等于 `VLLM_GPUS` 中 GPU 的数量

例如使用两张 rollout GPU：

```bash
TASK=webshop VLLM_GPUS=0,1 ./run_mprl_swift_rollout_server.sh
```

此时训练 GPU 必须改为其他设备，例如 `TRAIN_GPUS=2,3`。

### 9.2 Terminal 2：启用 Plan 训练（默认）

`PLANNING_ENABLED` 默认是 `true`，以下两种写法等价。

显式开启：

```bash
TASK=webshop \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=true \
./run_train_mprl_swift_server.sh
```

使用默认值：

```bash
TASK=webshop TRAIN_GPUS=1,2,3 ./run_train_mprl_swift_server.sh
```

切换任务时，server 和 trainer 的 `TASK` 必须一致：

```bash
TASK=alfworld \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=true \
./run_train_mprl_swift_server.sh
```

```bash
TASK=sciworld \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=true \
./run_train_mprl_swift_server.sh
```

### 9.3 Terminal 2：关闭 Plan 训练

关闭规划必须显式设置：

```bash
PLANNING_ENABLED=false
```

完整示例：

```bash
TASK=webshop \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=false \
EXPERIMENT_NAME=mprl-webshop-no-plan \
CHECKPOINT_DIR=/tmp/rllm-mprl-webshop-no-plan \
./run_train_mprl_swift_server.sh
```

ALFWorld：

```bash
TASK=alfworld \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=false \
EXPERIMENT_NAME=mprl-alfworld-no-plan \
CHECKPOINT_DIR=/tmp/rllm-mprl-alfworld-no-plan \
./run_train_mprl_swift_server.sh
```

ScienceWorld：

```bash
TASK=sciworld \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=false \
EXPERIMENT_NAME=mprl-sciworld-no-plan \
CHECKPOINT_DIR=/tmp/rllm-mprl-sciworld-no-plan \
./run_train_mprl_swift_server.sh
```

> `PLANNING_ENABLED=false` 只跳过初始 Meta-Plan 请求和上下文注入，不会关闭正常动作采样、GRPO、adapter 更新或 rollout 权重同步。

### 9.4 Plan 对照实验建议

比较 Plan 与 No-Plan 时至少保持以下配置一致：

- 相同基础模型和同一份初始 SFT adapter
- 相同 train/test indices
- 相同 `GROUP_SIZE`
- 相同 `MAX_STEPS`
- 相同采样 temperature/top-p
- 相同训练 epoch、学习率和 batch 配置

同时使用不同的：

- `EXPERIMENT_NAME`
- `CHECKPOINT_DIR`
- `LOG_DIR`（如需要完全隔离日志）

不要让 No-Plan 实验从已经经过 Plan-RL 更新的 checkpoint 恢复，否则不再是严格对照。

### 9.5 常用训练参数

可通过环境变量覆盖：

```bash
TASK=webshop \
TRAIN_GPUS=1,2,3 \
PLANNING_ENABLED=true \
PLANNING_MAX_TOKENS=1024 \
GROUP_SIZE=8 \
NUM_MINIBATCHES=8 \
TRAIN_BATCH_SIZE=1 \
MAX_PROMPT_LENGTH=6144 \
MAX_RESPONSE_LENGTH=4096 \
MAX_STEPS=12 \
N_PARALLEL_AGENTS=2 \
TRAIN_LIMIT=0 \
VAL_LIMIT=16 \
TOTAL_EPOCHS=2 \
EXPERIMENT_NAME=mprl-webshop-plan \
CHECKPOINT_DIR=/tmp/rllm-mprl-webshop-plan \
./run_train_mprl_swift_server.sh
```

任务默认值：

- WebShop：`MAX_STEPS=12`，`N_PARALLEL_AGENTS=2`
- ALFWorld：`MAX_STEPS=40`，`N_PARALLEL_AGENTS=2`
- ScienceWorld：`MAX_STEPS=60`，`N_PARALLEL_AGENTS=1`

显存不足时，优先按顺序调整：

1. 减少 `MAX_RESPONSE_LENGTH`。
2. 减少 `MAX_PROMPT_LENGTH`，但必须保证环境上下文仍能容纳。
3. 增加 `NUM_MINIBATCHES`。
4. 减少 `GROUP_SIZE`；这会改变 GRPO 每组样本数，应记录为算法配置变化。
5. 降低 rollout server 的 `VLLM_GPU_MEMORY_UTILIZATION`。

---

## 10. 只采样、不训练

`run_infer_demo.sh` 会启动普通 vLLM OpenAI server，加载任务 LoRA，然后调用 `mprl.run_interact` 采样。

```bash
cd /path/to/MPRL/rllm
TASK=webshop LIMIT=8 REPEAT_K=2 ./run_infer_demo.sh
```

`mprl.run_interact` 默认开启 Plan。直接调用 Python 入口时：

```bash
PYTHONPATH="$PWD:.." ../.venv-mprl311/bin/python -m mprl.run_interact \
  --task webshop \
  --base-url http://127.0.0.1:30000/v1 \
  --model-alias webshop-lora-adapter \
  --planning
```

关闭 Plan：

```bash
PYTHONPATH="$PWD:.." ../.venv-mprl311/bin/python -m mprl.run_interact \
  --task webshop \
  --base-url http://127.0.0.1:30000/v1 \
  --model-alias webshop-lora-adapter \
  --no-planning
```

训练脚本使用环境变量 `PLANNING_ENABLED=true|false`；采样 Python CLI 使用 `--planning|--no-planning`。不要混用这两套参数形式。

---

## 11. Benchmark 独立评测

运行前先检查 `maml/configs/*_eval_config.yaml` 中的基础模型和 adapter 路径。

ALFWorld：

```bash
./scripts/run_alfworld_eval.sh
```

WebShop：

```bash
./scripts/run_webshop_eval.sh
```

ScienceWorld：

```bash
./scripts/run_sciworld_eval.sh
```

日志写入仓库根目录的 `logs/`，并生成 `*.latest.log` 软链接。

这些脚本评测配置文件中指定的 adapter。第三阶段 checkpoint 若保存在其他位置，需要先更新对应 YAML 的 `adapter_name_or_path`。

---

## 12. 日志、验证和 checkpoint

Swift rollout server 日志：

```text
rllm/logs/mprl/<task>/vllm_<timestamp>.log
rllm/logs/mprl/<task>/latest.vllm.log
```

训练日志：

```text
rllm/logs/mprl/<task>/train_launcher_<timestamp>.log
rllm/logs/mprl/<task>/train_<timestamp>.out
rllm/logs/mprl/<task>/train_<timestamp>.err
rllm/logs/mprl/<task>/latest.train.launcher.log
rllm/logs/mprl/<task>/latest.train.out
rllm/logs/mprl/<task>/latest.train.err
```

默认训练行为：

- `val_before_train=true`：正式更新前先记录 step 0 验证。
- `test_freq=25`：每 25 个训练 batch 验证一次。
- `save_freq=2000`：每 2,000 个训练 batch 保存一次。
- 默认 checkpoint：`/tmp/rllm-mprl-<task>`

`/tmp` 不适合长期保存。正式实验应显式指定持久化目录：

```bash
CHECKPOINT_DIR=/mnt/hdfs/<user>/mprl-checkpoints/webshop-plan
```

---

## 13. 测试

从 `rllm/` 目录执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
../.venv-mprl311/bin/python -m pytest \
tests/mprl/test_mprl_flow.py -q
```

设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 是为了避免环境中与本项目无关的第三方 pytest plugin 自动加载及其依赖冲突。

测试覆盖：

- 三任务数据和 prompt 映射
- Meta-Plan context-only 注入
- 显式关闭 Plan
- 规划请求与动作请求的 batching 隔离
- LoRA adapter 可训练加载
- gradient checkpointing 的计算图
- ScienceWorld 日志兼容修复
- DDP rank 使用固定 minibatch collective 数

---

## 14. 常见问题

### 14.1 `Weight update group already initialized`

上一次训练异常退出后，vLLM server 内仍保留旧 Swift/NCCL communicator。新的训练进程无法复用该 communicator。

处理方式：

1. 停止当前训练。
2. 停止并重启 `run_mprl_swift_rollout_server.sh`。
3. 等待 health check 成功。
4. 重新启动训练。

### 14.2 `Port 8000 is already in use`

检查占用：

```bash
ss -lptn "sport = :8000"
```

停止旧 server，或同时为 server 和 trainer 指定新端口：

```bash
VLLM_PORT=8001 ./run_mprl_swift_rollout_server.sh
VLLM_PORT=8001 ./run_train_mprl_swift_server.sh
```

### 14.3 WebShop 找不到 `libjvm.so`

不要使用旧的：

```text
/opt/tiger/jdk/jdk1.8/lib/server/libjvm.so
```

默认应为：

```text
/opt/tiger/jdk/jdk11/lib/server/libjvm.so
```

### 14.4 `CUDA out of memory. Tried to allocate more than 1EB`

1EB 不是正常模型内存申请，通常说明 DDP collective 次序错位，而不是 sequence 太长。本仓库的 Swift trainer 已通过固定各 rank 的 minibatch 数和固定大小同步状态张量进行修复。

普通的 GiB 级 OOM 才应按照第 9.5 节调整长度、minibatch、group size 或 vLLM 显存比例。

### 14.5 训练失败后能否只重启 trainer

- 如果失败发生在模型/环境初始化阶段，尚未初始化 Swift weight communicator：通常只需重启 trainer。
- 如果已经发生过权重同步，或 server 日志出现 communicator 错误：必须重启 rollout server。

### 14.6 Gym 0.24 warning

当前环境会输出 Gym 0.24 的兼容性 warning。它不是本项目已知训练失败的直接原因。不要仅为消除 warning 单独升级 Gym；ALFWorld/WebShop 的旧环境 API 可能依赖当前版本，升级前需要完整回归测试。

---

## 15. 推荐的完整实验顺序

1. 检查基础模型、三类数据和 benchmark 环境资产。
2. 在三个 Meta-Plan 数据集上运行 MAML。
3. 用 Meta-Plan 测试集检查 workflow 格式和 ROUGE/BLEU。
4. 从同一 MAML 产物分别进行 ALFWorld、WebShop、ScienceWorld SFT。
5. 使用独立 benchmark 脚本评测三个 SFT adapter。
6. 对每个任务单独启动 Swift rollout server。
7. 先运行 `PLANNING_ENABLED=true` 的第三阶段 RL。
8. 使用相同初始 SFT adapter 和超参数运行 `PLANNING_ENABLED=false` 对照实验。
9. 比较 reward、success rate、训练稳定性、平均交互步数和推理开销。
10. 将 checkpoint 保存到持久化目录，并保留对应配置、日志和 WandB run。

