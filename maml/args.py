from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union
import yaml
import os

import torch
import transformers
from transformers import Seq2SeqTrainingArguments, HfArgumentParser, GenerationConfig
from transformers.utils import is_torch_cuda_available

from .logger import logging


logger = logging.get_logger(__name__)


@dataclass
class DataArgs:
    r"""Args pertaining to what data we are going to input our model for training and evaluation."""
    
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The file name (with format suffix, e.g. .json) of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The file name (with format suffix, e.g. .json) of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data/metaplan/train",
        metadata={"help": "Path to the folder containing the datasets for MAML."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    max_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["interleave_under", "interleave_over"] = field(
        default="interleave_under",
        metadata={"help": "Strategy to use in dataset mixing (undersampling/oversampling) (Interleave is preferred for MAML)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    
    def __post_init__(self):
        if self.dataset is not None:
            self.dataset = [x.strip() for x in self.dataset.split(",")]
        
        if self.eval_dataset is not None:
            self.eval_dataset = [x.strip() for x in self.eval_dataset.split(",")]
        
        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")
        
        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")
        
        if self.interleave_probs is not None:
            self.interleave_probs = [float(x) for x in self.interleave_probs.split(",")]
            
            if self.dataset is not None and len(self.interleave_probs) != len(self.dataset):
                raise ValueError("The number of interleave probabilities must match the number of datasets.")
            
            if self.eval_dataset is not None and len(self.interleave_probs) != len(self.eval_dataset):
                raise ValueError("The number of interleave probabilities must match the number of eval datasets.")
        
        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")
            
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    
@dataclass
class ModelArgs:
    r"""Args pertaining to what model we are going to use for MAML."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    # Do not specify these args, they are derived from other args
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    device_map: Optional[Union[str, dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    
    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    

@dataclass
class TrainingArgs(Seq2SeqTrainingArguments):
    r"""Args pertaining to training configuration."""
    
    def __post_init__(self):
        super().__post_init__()
        
        
@dataclass
class FinetuningArgs:
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    training_type: Literal["lora", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )        
    
    def __post_init__(self):
        self.lora_target: list[str] = [x.strip() for x in self.lora_target.split(",")]
        self.lora_alpha = self.lora_alpha if self.lora_alpha is not None else self.lora_rank * 2
        assert self.training_type in ["lora", "full"], "Invalid fine-tuning method."


@dataclass
class GenerationArgs:
    r"""Args pertaining to generation configuration."""
    
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.9,
        metadata={
            "help": (
                "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
            )
        },
    )
    top_k: int = field(
        default=20,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether or not to remove special tokens in the decoding."},
    )
    
    def to_dict(self, obey_generation_config: bool = False) -> dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)

        if obey_generation_config:
            generation_config = GenerationConfig()
            for key in list(args.keys()):
                if not hasattr(generation_config, key):
                    args.pop(key)

        return args


def read_training_args(args) -> tuple[DataArgs, ModelArgs, GenerationArgs, FinetuningArgs, TrainingArgs]:
    r"""Read and parse the configuration file for training."""
    with open(args.config, "r", encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        
    parser = HfArgumentParser([DataArgs, ModelArgs, GenerationArgs, FinetuningArgs, TrainingArgs])
    data_args, model_args, generation_args, finetuning_args, training_args = parser.parse_dict(config_dict)
    
    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")
    
    if model_args.adapter_name_or_path is not None and finetuning_args.training_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")
    
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    if finetuning_args.training_type == "lora":
        training_args.label_names = training_args.label_names or ["labels"]
    
    if training_args.predict_with_generate:
        if data_args.eval_dataset is None:
            raise ValueError("Cannot use `predict_with_generate` if there is no evaluation dataset.")
        
    # set model args derived from training and data args
    if training_args.bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
    if is_torch_cuda_available():
        model_args.device_map = "auto"
    else:
        model_args.device_map = {"": torch.device("cpu")}
    model_args.model_max_length = data_args.cutoff_len
    
    logger.info(
        f"Process rank: {training_args.process_index}, "
        f"world size: {training_args.world_size}, device: {training_args.device}, "
        f"compute dtype: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(training_args.seed)
    
    return data_args, model_args, generation_args, finetuning_args, training_args


def read_metaplan_eval_args(args) -> tuple[DataArgs, ModelArgs, GenerationArgs, FinetuningArgs, TrainingArgs]:
    r"""Read and parse the configuration file for Meta-Plan evaluation."""
    with open(args.config, "r", encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    parser = HfArgumentParser([DataArgs, ModelArgs, GenerationArgs, FinetuningArgs, TrainingArgs])
    data_args, model_args, generation_args, finetuning_args, training_args = parser.parse_dict(config_dict)
    
    if model_args.adapter_name_or_path is not None and finetuning_args.training_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")
    
    if is_torch_cuda_available():
        model_args.device_map = "auto"
    else:
        model_args.device_map = {"": torch.device("cpu")}
    model_args.model_max_length = data_args.cutoff_len
    
    return data_args, model_args, generation_args, finetuning_args, training_args


def read_specify_task_eval_args(args) -> tuple[DataArgs, ModelArgs, GenerationArgs, FinetuningArgs]:
    r"""Read and parse the configuration file for specify-task evaluation. (No need for loading Datasets)"""
    with open(args.config, "r", encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    parser = HfArgumentParser([DataArgs, ModelArgs, GenerationArgs, FinetuningArgs])
    data_args, model_args, generation_args, finetuning_args = parser.parse_dict(config_dict)
    
    if model_args.adapter_name_or_path is not None and finetuning_args.training_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")
    
    if is_torch_cuda_available():
        model_args.device_map = "auto"
    else:
        model_args.device_map = {"": torch.device("cpu")}
    model_args.model_max_length = data_args.cutoff_len
    
    return data_args, model_args, generation_args, finetuning_args