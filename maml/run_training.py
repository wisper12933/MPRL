import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import jieba
import numpy as np
import matplotlib.pyplot as plt
from transformers.trainer import TRAINER_STATE_NAME
from rouge_chinese import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel

from .logger import logging
from .args import read_training_args
from .trainer import MAMLTrainer
from .data_loader import get_dataset, get_template_and_fix_tokenizer, MAMLDataCollatorForSeq2Seq, IGNORE_INDEX
from .model_loader import load_tokenizer, load_model


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, EvalPrediction


logger = logging.get_logger(__name__)


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"
    
    def _numpify(self, inputs: Union["NDArray", "torch.Tensor"]) -> "NDArray":
        r"""Cast a torch tensor or a numpy array to a numpy array."""
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu()
            if inputs.dtype == torch.bfloat16: 
                inputs = inputs.to(torch.float32)

            inputs = inputs.numpy()

        return inputs

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = self._numpify(eval_preds.predictions), self._numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


def plot_loss(save_dictionary: str, keys: list[str] = ["loss"]) -> None:
    r"""Plot loss curves and saves the image."""
    
    def smooth(scalars: list[float]) -> list[float]:
        r"""EMA implementation according to TensorBoard."""
        if len(scalars) == 0:
            return []

        last = scalars[0]
        smoothed = []
        weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
        for next_val in scalars:
            smoothed_val = last * weight + (1 - weight) * next_val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title(f"training {key} of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


def main():
    parser = argparse.ArgumentParser(description="MAML Training Main Function")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file containing DataArgs, ModelArgs, and TrainArgs. For example: configs/config.yaml"
    )
    args = parser.parse_args()
    
    data_args, model_args, generation_args, finetuning_args, training_args = read_training_args(args)
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, tokenizer)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    if dataset_module.get("train_dataset") is None:
        raise ValueError("Trainer: training requires a train_dataset.")
    
    data_collator = MAMLDataCollatorForSeq2Seq(
        model=model if not training_args.predict_with_generate else None,
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        compute_dtype=model_args.compute_dtype,
    )
    
    metric_module = {}
    if training_args.do_eval:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer)
        
    gen_kwargs = generation_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    trainer = MAMLTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **metric_module,
    )
    
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += sum(
                [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
            )
        else:
            keys += ["eval_loss", "eval_accuracy"]
        plot_loss(training_args.output_dir, keys)
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    card_kwargs = {
        "tasks": "text-generation: MAML metaplan",
        "model_name": model_args.model_name_or_path,
        "tags": ["MAML", "metaplan"],
    }
    
    if data_args.dataset is not None:
        card_kwargs["dataset"] = [dataset.split('.')[0] for dataset in data_args.dataset]
    
    trainer.create_model_card(license="other", **card_kwargs)
    

if __name__ == "__main__":
    main()
    