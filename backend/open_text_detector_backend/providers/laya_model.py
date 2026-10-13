"""Reviewed Laya inference architecture (Apache-2.0; NandhaKishorM/laya v0.3.20).

Matches upstream parameter names; loads only local JSON and safetensors, never Hub code.
The unused action/escalation head is retained for strict checkpoint validation.
"""
import json
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.initialization import no_init_weights
from safetensors.torch import load_file


class LayaModel(nn.Module):
    def __init__(self, directory):
        super().__init__()
        cfg = json.loads((directory / 'rl_agent_config.json').read_text())
        ecfg = AutoConfig.from_pretrained(directory / 'encoder', local_files_only=True, trust_remote_code=False)
        ecfg.reference_compile = False
        if ecfg.model_type != 'modernbert' or ecfg.hidden_size != 1024 or cfg['head_layers'] != 2:
            raise ValueError('Unsupported Laya architecture')
        with no_init_weights():
            self.encoder = AutoModel.from_config(ecfg, attn_implementation='sdpa', trust_remote_code=False)
            d = ecfg.hidden_size
            self.head = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d, d // 64, 4 * d, .1, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))
            self.register_buffer('temperature', torch.ones(3))
        self.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        self.eval()

    def forward(self, ids, lengths, markers):
        mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        h = h + self.type_emb.weight[0]  # upstream choice question type
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=~mask)
        h = h[torch.arange(len(ids), device=ids.device)[:, None], markers]
        return self.scorer(h).squeeze(-1).float()
