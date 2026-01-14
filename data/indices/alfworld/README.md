## 加载特定的测试/训练实例

```python
target_game_file = "..."

with open("envs/alfworld/base_config.yaml") as reader:
    config = yaml.safe_load(reader)
# train_eval: 训练使用 train，测试使用 eval_out_of_distribution
env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="eval_out_of_distribution")

env.game_files = [target_game_file]

env = env.init_env(batch_size=1)
```