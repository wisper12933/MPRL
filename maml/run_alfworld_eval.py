import re
import sys
import yaml
import argparse

import torch
import alfworld

from .args import read_specify_task_eval_args
from .model_loader import load_tokenizer, load_model
from .data_loader import get_template_and_fix_tokenizer


with open("/mnt/home/user28/MPRL/data/instructions/alfworld_inst.txt", 'r') as f:
    BASE_PROMPT = f.read()    

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


def alfworld_run(env, messages, template, tokenizer, model, gen_kwargs):
    r"""Run Alfworld evaluation loop"""
    
    def extract_action(s: str):
        """Extract action from model output string"""
        s = s.strip()
        pattern = re.compile(r"Action: (.*)")
        matches = re.findall(pattern, s)
        if not matches:
            return ""
        return matches[0].strip()
    
    def process_ob(ob):
        """Process observation string"""
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ')+2:]    
        return ob
    
    # print initial messages
    for message in messages:
        print(message["content"] + '\n')
    sys.stdout.flush()
    
    # setting max_error_step for early stopping when invalid actions repeatedly occur
    curr_error_step, max_error_step = 0, 6
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
        else:
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            observation = f"Observation: {observation}"
            
            if "Nothing happens" in observation:
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
    
    return 0
    

def main():
    parser = argparse.ArgumentParser(description="Specify-task Evaluation Main Function")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing DataArgs, ModelArgs, and TrainArgs. For example: configs/config.yaml"
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
    
    # load alfworld env
    with open("alfworld/base_config.yaml") as reader:
        config = yaml.safe_load(reader)
    
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)
    
    cnts = [0] * 6  # cnts stores total task num
    rs = [0] * 6  # rs stores rewards derived from each task
    
    # start evaluation on out-of-distribution tasks
    for _ in range(134):
        ob, info = env.reset()
        print(f'Show: ob={ob}')
        print(f'Show: info={info}')
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(name)
        print('*' * 50 + f'\nBegin Task {_ + 1}\n' + '*' * 50)
        
        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                print(k, v)
                messages = [
                    {"role": "user", "content": BASE_PROMPT},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": ob}
                ]
                r = alfworld_run(env, messages, template, tokenizer, model, gen_kwargs)
                
                cnts[i] += 1
                rs[i] += r
                break
            
        print(_+1, 'Reward', r, 'Rewards', rs, 'Counts', cnts, 'Avg. Success Rate', sum(rs) / sum(cnts))
        print('------------\n')
    

if __name__ == "__main__":
    main()