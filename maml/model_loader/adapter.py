from typing import TYPE_CHECKING

import torch
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from ..logger import logging
from .utils import find_all_linear_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel
    
    from ..args import ModelArgs, TrainingArgs


logger = logging.get_logger(__name__)


def _setup_full_tuning(
    model: "PreTrainedModel",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return
    
    logger.info_rank0("Tuning method: Full")
    if cast_trainable_params_to_fp32:
        for param in model.parameters():
            param.data = param.data.to(torch.float32)
    

def _setup_lora_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArgs",
    finetuning_args: "TrainingArgs",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        logger.info_rank0("Tuning method: LoRA")
    
    if model_args.adapter_name_or_path is not None:
        # inference or training with pre-trained adapter
        adapter_to_merge = model_args.adapter_name_or_path
        
        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
        }
        
        model = PeftModel.from_pretrained(model, adapter_to_merge, is_trainable=is_trainable, **init_kwargs)
        model = model.merge_and_unload()
        
        logger.info_rank0(f"Loaded adapter from {adapter_to_merge}.")
    
    if is_trainable and model_args.adapter_name_or_path is None:
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model)
        else:
            target_modules = finetuning_args.lora_target
        
        lora_config = LoraConfig(
            r=finetuning_args.lora_rank,
            target_modules=target_modules,
            lora_alpha=finetuning_args.lora_alpha,
            lora_dropout=finetuning_args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
    
    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
    
    return model


def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArgs",
    training_args: "TrainingArgs",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True
    
    if training_args.training_type == "full":
        _setup_full_tuning(model, is_trainable, cast_trainable_params_to_fp32)
    elif training_args.training_type == "lora":
        model = _setup_lora_tuning(model, model_args, training_args, is_trainable, cast_trainable_params_to_fp32)
    else:
        raise ValueError(f"Unsupported training type: {training_args.training_type}")
    
    return model
    