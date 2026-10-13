"""Merged Qwen checkpoint with EditLens's original LayerNorm + Linear head.

Adapted from pangramlabs/EditLens scripts/train.py (CC BY-NC-SA 4.0)
and DarrenJiaImbue/editlens-qwen3-4b-merged-v3's model card.
Install the correct head BEFORE loading, so all checkpoint tensors are loaded
strictly and no randomly initialized classifier can silently serve predictions.
"""
import torch
from transformers import Qwen3ForSequenceClassification


class NormedLinear(torch.nn.Module):
    def __init__(self, hidden_size, labels):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, labels, bias=False)

    def forward(self, hidden):
        return self.linear(self.norm(hidden))


class EditLensQwen(Qwen3ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.score = NormedLinear(config.hidden_size, config.num_labels)


def load_merged_qwen(path, dtype):
    model, info = EditLensQwen.from_pretrained(
        path, dtype=dtype, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Checkpoint tensors do not match the reviewed EditLens architecture")
    model.config.use_cache = False
    return model
