import os
import sys
from typing import Optional, Any, Union

import torch
import torch.nn as nn
from peft import PeftModel
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import Seq2SeqTrainer
from typing_extensions import override

from .logger import logging


logger = logging.get_logger(__name__)


class MAMLTrainer(Seq2SeqTrainer):
    r"""Trainer class to implement MAML training.
    
    inner_lr: Learning rate for the inner loop. 
    inner_steps: Number of inner loop steps.
    support_size: Number of samples in the support set for inner loop.
    query_size: Number of samples in the query set for outer loop.
    gen_kwargs: Generation kwargs for prediction step.
    
    FOMAML √ 
    MAML √
    """
    def __init__(
        self, 
        inner_lr: float = 1e-4, 
        inner_steps: int = 1,
        support_size: int = 2,
        query_size: int = 2,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.support_size = support_size
        self.query_size = query_size
        
        self._gen_kwargs = gen_kwargs
        
    def _maml_training_step(self, model, batch):
        r"""MAML training step implementation."""
        # split batch into support and query sets
        
        total_size = self.support_size + self.query_size
        if total_size != batch['input_ids'].size(0):
            raise ValueError(f"Batch size {batch['input_ids'].size(0)} does not match maml support_size + query_size = {total_size}")
            
        support_batch = {}
        query_batch = {}
        
        for key, value in batch.items():
            if key in ['input_ids', 'attention_mask', 'labels']:
                support_batch[key] = value[:self.support_size]
                query_batch[key] = value[self.support_size:total_size]
        
        return self._functional_maml_step(model, support_batch, query_batch)
    
    def _functional_maml_step(self, model, support_batch, query_batch):
        """Unified MAML step using functional_call for both FOMAML and MAML."""
        # prepare functional parameters
        params = dict(model.named_parameters()) 
        buffers = dict(model.named_buffers())
        functional_params = {**params, **buffers}
        
        trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        
        inner_losses = []
        
        sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
        
        with sdpa_ctx:
            for step in range(self.inner_steps):
                # Inner loop
                outputs = functional_call(model, functional_params, args=(), kwargs=support_batch, tie_weights=False)
                loss = outputs.loss
                
                if torch.isnan(loss):
                    logger.warning(f"Inner loss is NaN at step {step}, skipping inner update")
                    continue
                
                inner_losses.append(loss)
                
                trainable_params = [functional_params[n] for n in trainable_param_names]
                
                grads = torch.autograd.grad(
                    loss, 
                    trainable_params,
                    create_graph=True, 
                    allow_unused=True
                )
                
                updated_params = functional_params.copy() 
                for name, grad in zip(trainable_param_names, grads):
                    if grad is not None:
                        # gradient clipping
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            logger.warning_rank0(f"NaN or Inf detected in gradients for parameter {name}. Skipping update.")
                            grad = torch.zeros_like(grad)
                        else:
                            grad = torch.clamp(grad, -1.0, 1.0)
                        
                        # theta' = theta - lr * grad
                        updated_params[name] = functional_params[name] - self.inner_lr * grad
                
                functional_params = updated_params
            
        # Outer loop
        query_outputs = functional_call(model, functional_params, args=(), kwargs=query_batch, tie_weights=False)
        meta_loss = query_outputs.loss
        
        return meta_loss, inner_losses
    
    def training_step(self, model: nn.Module, inputs: dict[str, torch.Tensor], num_items_in_batch: int = None) -> torch.Tensor:
        r"""Rewrite the training step for MAML."""
        
        meta_loss, inner_losses = self._maml_training_step(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            meta_loss = meta_loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(meta_loss)
        
        self._inner_loss = torch.stack(inner_losses).mean().detach().item() if inner_losses else 0.0
        sys.stdout.flush()
        
        return meta_loss.detach()

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if hasattr(self, '_inner_loss'):
            logs['inner_loss'] = self._inner_loss
        super().log(logs, *args, **kwargs)
    
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels
