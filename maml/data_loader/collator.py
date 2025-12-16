from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq


@dataclass
class MAMLDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""
    
    compute_dtype: "torch.dtype" = torch.float32
    
    def __post_init__(self):

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        
        features: dict[str, torch.Tensor] = super().__call__(features)
        for key, value in features.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)
        
        return features
