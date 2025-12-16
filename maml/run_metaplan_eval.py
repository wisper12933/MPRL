import argparse

import torch
import jieba
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from rouge_chinese import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .args import read_metaplan_eval_args
from .data_loader import get_dataset, get_template_and_fix_tokenizer, MAMLDataCollatorForSeq2Seq, IGNORE_INDEX
from .model_loader import load_tokenizer, load_model

def main():
    parser = argparse.ArgumentParser(description="Meta-plan Evaluation Main Function")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing DataArgs, ModelArgs, and TrainArgs. For example: configs/config.yaml"
    )
    args = parser.parse_args()
    
    data_args, model_args, generation_args, finetuning_args, training_args = read_metaplan_eval_args(args)
    
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, tokenizer)
    model = load_model(tokenizer, model_args, finetuning_args)
    
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id >= len(tokenizer):
        print(f"Resetting pad_token_id from {tokenizer.pad_token_id} to {tokenizer.eos_token_id}")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Total number of evaluation samples: {len(dataset_module['eval_dataset'])}")
    
    tokenizer.padding_side = "left"
    # get data loader
    data_collator = MAMLDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        compute_dtype=model_args.compute_dtype,
    )
    
    data_loader = DataLoader(
        dataset_module['eval_dataset'],
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
    )
    
    gen_kwargs = generation_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    decoded_preds = []
    decoded_labels = []
    
    device = model.device
    
    # inference
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        labels = batch['labels'] 

        input_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        generated_tokens = outputs[:, input_len:]
        batch_pred_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        batch_label_texts = []
        for lbl in labels:
            valid_labels = [token_id.item() for token_id in lbl if token_id.item() != -100]
            batch_label_texts.append(tokenizer.decode(valid_labels, skip_special_tokens=True))
        
        decoded_preds.extend(batch_pred_texts)
        decoded_labels.extend(batch_label_texts)
        
    # calculate rouge and bleu scores
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": [],
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = list(jieba.cut(pred))
        label_tokens = list(jieba.cut(label))
        
        if len(" ".join(pred_tokens).split()) == 0 or len(" ".join(label_tokens).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(pred_tokens), " ".join(label_tokens))
            result = scores[0]
        
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))
    
    avg_scores = {k: float(np.mean(v)) for k, v in score_dict.items()}
    print("Evaluation Results:")
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")
    

if __name__ == "__main__":
    main()
    