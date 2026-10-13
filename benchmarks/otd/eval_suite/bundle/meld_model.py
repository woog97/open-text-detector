"""Reviewed MELD v5/v8 evidence heads, adapted from the MIT-licensed upstream release.

Source: https://huggingface.co/anon-review-meld-2026/meld
The token margins are evidence, not supervised three-way provenance labels.
"""

import json
from pathlib import Path
import torch
from safetensors.torch import load
from transformers import AutoConfig, AutoModel
from transformers.initialization import no_init_weights


class MeldModel(torch.nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        self.cfg = json.loads((directory / "meld_config.json").read_text())
        if self.cfg.get("version") not in {"v5", "v8"}:
            raise ValueError("Expected a reviewed MELD v5 or v8 architecture")
        rank, hidden = self.cfg["style_rank"], self.cfg["backbone_hidden_size"]
        # All parameters are supplied by the strictly validated checkpoint below.
        # Avoid spending time generating random weights that are immediately discarded.
        with no_init_weights():
            self.backbone = AutoModel.from_config(
                AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False),
                attn_implementation="sdpa",
                trust_remote_code=False,
            )
        self.style_proj = torch.nn.Linear(hidden, rank, bias=False)
        self.style_ln = torch.nn.LayerNorm(rank)
        self.human_anchors = torch.nn.Parameter(torch.zeros(self.cfg["n_human_anchors"], rank))
        self.family_protos = torch.nn.Parameter(torch.zeros(self.cfg["n_families"], rank))
        self.family_bias = torch.nn.Parameter(torch.zeros(self.cfg["n_families"]))
        self.log_tau = torch.nn.Parameter(torch.zeros(()))
        self.op_protos = torch.nn.Parameter(torch.zeros(self.cfg["n_ops"], rank))
        self.op_bias = torch.nn.Parameter(torch.zeros(self.cfg["n_ops"]))
        # Fail closed on any missing or unexpected tensors, including the backbone.
        self.load_state_dict(load((directory / "model.safetensors").read_bytes()), strict=True)
        self.eval()

    def token_scores(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        style = self.style_ln(self.style_proj(hidden))
        tau = self.log_tau.clamp(-4, 4).exp()

        def distance(prototypes):
            return (
                (style * style).sum(-1, keepdim=True)
                - 2 * style @ prototypes.t()
                + (prototypes * prototypes).sum(-1).view(1, 1, -1)
            )

        human_logits = -tau * distance(self.human_anchors)
        human = (human_logits.max(-1, keepdim=True).values if self.cfg.get("null_reduction") == "max"
                 else torch.logsumexp(human_logits, -1, keepdim=True))
        family = -tau * distance(self.family_protos) + self.family_bias.view(1, 1, -1)
        scale = self.cfg["tau_agg"]
        return scale * torch.logsumexp((family - human).clamp(-30, 30) / scale, dim=-1)
