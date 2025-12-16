import re
import sys
import json
import argparse
import os

import torch
from scienceworld import ScienceWorldEnv

from .args import read_specify_task_eval_args
from .model_loader import load_tokenizer, load_model
from .data_loader import get_template_and_fix_tokenizer


with open("/mnt/home/user28/MPRL/data/instructions/sciworld_inst.txt", 'r') as f:
    BASE_PROMPT = f.read()    

env_jar_path = "data/sciworld/scienceworld.jar"

    
def sciworld_step_patch():
    r"""Patch ScienceWorldEnv step function"""
    def step(self, inputStr:str):
        observation = self.server.step(inputStr)
        raw_score = self.server.getScore()
        score = int(round(100 * raw_score))        # Convert from 0-1 to 0-100
        isCompleted = self.server.getCompleted()
        numMoves = self.getNumMoves()

        # Calculate reward
        reward = score - self.lastStepScore         # Calculate reward (delta score) for this step
        self.lastStepScore = score                  # Store current score for reward calculation on the next step


        # If the number of moves exceeds the environment step limit, then set isCompleted to be true
        if (numMoves > self.envStepLimit):
            isCompleted = True

        # New: Handle this in the API rather than the agent -- if the score is less than zero, then set the isCompleted flag to true.
        if (score < 0):
            isCompleted = True
        
        taskDesc = self.taskdescription()
        taskDesc = taskDesc.split('Task Description:\n')[1].strip()

        # Mirror of Jericho API
        infos = {
            'moves': numMoves,
            'raw_score': raw_score,
            'score': score,
            'reward': reward,
            'look': self.look(),
            'inv': self.inventory(),
            'taskDesc': taskDesc,
            'valid': self.getValidActionObjectCombinations(),
            'variationIdx': self.variationIdx,
            'taskName': self.taskName,
            'simplificationStr': self.simplificationStr,
        }

        return observation, reward, isCompleted, infos
    
    ScienceWorldEnv.step = step
    print("Patched ScienceWorldEnv.step function.")
    

def sciworld_run(env, messages, template, tokenizer, model, gen_kwargs):
    r"""Run SciWorld evaluation loop"""
    
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
    reward, curr_error_step, max_error_step = 0, 0, 10
    curr_invalid_step, max_invalid_step = 0, 5
    for _ in range(40):
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
            curr_invalid_step += 1
            if curr_invalid_step >= max_invalid_step:
                done = True
        else:
            observation, _, done, info = env.step(action)
            observation = f"Observation: {observation}"
            reward = max(reward, info["raw_score"])
            curr_invalid_step = 0
            
            if "No known action matches that input" in observation:
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
    parser = argparse.ArgumentParser(description="SciWorld Evaluation Main Function")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing DataArgs, ModelArgs, and TrainArgs."
    )
    parser.add_argument(
        "--test_idx_path",
        type=str,
        required=True,
        help="Path to the indices JSON file for the selected test tasks."
    )
    args = parser.parse_args()
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
    
    # load sciworld env
    sciworld_step_patch()
    env = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_jar_path), envStepLimit=200)
    
    # load test tasks
    if not os.path.exists(args.test_idx_path):
        raise FileNotFoundError(f"Test indices file not found at: {args.test_idx_path}")
        
    with open(args.test_idx_path, 'r') as f:
        test_ids = json.load(f)
            
    rewards = []
    
    print(f"Loaded {len(test_ids)} tasks from {args.test_idx_path}")

    # start evaluation
    for task_idx, (task_name, variation_idx) in enumerate(test_ids):
        print('*' * 50 + f'\nBegin Task {task_idx + 1}: {task_name} (Var {variation_idx})\n' + '*' * 50)
        
        env.load(task_name, variationIdx=variation_idx, simplificationStr="easy", generateGoldPath=False)
        ob, info = env.reset()
        
        key = task_name + str(variation_idx)
        messages = [
            {"role": "user", "content": BASE_PROMPT},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": info['taskDesc']},
        ]
        
        r = sciworld_run(env, messages, template, tokenizer, model, gen_kwargs)
        rewards.append(r)
        success_cnt = rewards.count(1.0)
            
        print(f'{task_idx+1}, Reward: {r}, Avg. Reward: {sum(rewards) / len(rewards)}, Avg. Success Rate: {success_cnt / len(rewards)}')
        print('------------\n')


if __name__ == "__main__":
    main()