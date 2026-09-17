import os
import re
import sys
import json
import argparse

import torch
from envs.webshop.web_agent_site.envs import WebAgentTextEnv

from .args import read_specify_task_eval_args
from .model_loader import load_tokenizer, load_model
from .data_loader import get_template_and_fix_tokenizer


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "maml", "configs", "webshop_eval_config.yaml")
DEFAULT_PROMPT_PATH = os.path.join(REPO_ROOT, "data", "instructions", "webshop_inst.txt")
DEFAULT_TEST_IDX_PATH = os.path.join(REPO_ROOT, "data", "indices", "webshop", "test_indices.json")


### Evaluation loop function
def webshop_run(env, task, messages, template, tokenizer, model, gen_kwargs):
    r"""Run Webshop evaluation loop"""
    
    def extract_action(s: str):
        """Extract action from model output string"""
        s = s.strip()
        pattern = re.compile(r"Action: (.*)")
        matches = re.findall(pattern, s)
        if not matches:
            return ""
        return matches[0].strip()
    
    # print initial messages
    for message in messages:
        print(message["content"] + '\n')
    sys.stdout.flush()
    
    # setting a max reward to track best performance, and max_error_step for early stopping when invalid actions repeatedly occur
    reward, curr_error_step, max_error_step = 0, 0, 3
    for _ in range(12):
        input_ids = template.encode_inputs(tokenizer, messages)
        input_ids = torch.tensor([input_ids]).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        
        output_ids = output_ids[0][input_len:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        action = extract_action(output_text)
        # update observation, reward, done
        if action == "":
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            done = False
        else:
            try:
                observation, reward, done, info = env.step(action=action)
                observation = f"Observation: {observation}"
            except AssertionError:
                observation = "Observation: Invalid action!"
                done = False

            if "Invalid action!" in observation:
                curr_error_step += 1
                if curr_error_step >= max_error_step:
                    done = True
            else:
                curr_error_step = 0
        
        print(f"{output_text}\n###{observation}\n")
        sys.stdout.flush()
        
        if done:
            return reward
        
        # append new user and assistant messages
        messages.extend([
            {"role": "assistant", "content": output_text},
            {"role": "user", "content": observation}
        ])
    
    return reward    
    

def main():
    parser = argparse.ArgumentParser(description="WebShop Evaluation Main Function")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file containing DataArgs, ModelArgs, and TrainArgs."
    )
    parser.add_argument(
        "--test_idx_path",
        type=str,
        default=DEFAULT_TEST_IDX_PATH,
        help="Path to the indices JSON file for the selected test tasks."
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the base prompt file."
    )
    args = parser.parse_args()
    with open(args.prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    # read args
    data_args, model_args, generation_args, finetuning_args = read_specify_task_eval_args(args)
    # load model and tokenizer
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)
    
    tokenizer.padding_side = "left"
    gen_kwargs = generation_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    # load webshop env
    env = WebAgentTextEnv(observation_mode="text", human_goals=True)
    
    # load test tasks
    if not os.path.exists(args.test_idx_path):
        raise FileNotFoundError(f"Test indices file not found at: {args.test_idx_path}")
    
    with open(args.test_idx_path, 'r') as f:
        test_ids = json.load(f)
    
    rewards, err_cnt = [], 0
    
    print(f"Loaded {len(test_ids)} tasks from {args.test_idx_path}")
    
    # start evaluation
    for task_idx, task in enumerate(test_ids):
        print('*' * 50 + f'\nBegin Task {task_idx + 1}: WebShop {task}\n' + '*' * 50)
        
        env.reset(task)
        
        messages = [
            {"role": "user", "content": base_prompt},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": env.observation},
        ]
        
        try:
            r = webshop_run(env, task, messages, template, tokenizer, model, gen_kwargs)
        except AssertionError:
            r = 0.0
            err_cnt += 1
        rewards.append(r)
        avg_rewards, avg_success_rate, avg_err_rate = sum(rewards) / len(rewards), rewards.count(1.0) / len(rewards), err_cnt / len(rewards)
        
        print(f"id:{task_idx + 1}, score:{avg_rewards}, success rate:{avg_success_rate}, false rate:{avg_err_rate}")
        print('------------\n')
    

if __name__ == "__main__":
    main()