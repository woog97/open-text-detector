"""Packed forwards for causal (Repeat2) backbones: a micro-batch's rows run as one sequence, so padding costs nothing.

Rows are right-padded by collate. Their real tokens are concatenated into a (1, sum of lengths) sequence with per-row
position ids and cu_seq_lens. The gated-delta-net layers restart their recurrence and convolution at each boundary
(cu_seq_lens_q; the convolution needs the ccdshim causal_conv1d, since the CUDA build ignores it). The full-attention
layers use 'packed_sdpa': rows are padded back only inside attention and run with SDPA's causal flash path (a packed
block-diagonal mask would force a dense T x T mask). Hidden states are scattered back to the padded layout, so heads
and losses are unchanged.
"""
import torch
import modeling_sweep
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

NAME = 'packed_sdpa'
CALLS = {'attention': 0, 'packed_forward': 0}


def _rows_cols(cu):
    lens = cu[1:] - cu[:-1]; rows = torch.repeat_interleave(torch.arange(len(lens), device=cu.device), lens)
    return rows, torch.arange(int(cu[-1]), device=cu.device) - cu[rows].long(), int(lens.max())


def packed_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    cu = kwargs.get('cu_seq_lens_q')
    assert cu is not None and attention_mask is None and query.shape[0] == 1, 'packed_sdpa needs a packed (1, T) input with cu_seq_lens_q'
    rows, cols, L = _rows_cols(cu); B = len(cu) - 1; CALLS['attention'] += 1

    def pad(x):  # (1, H, T, D) -> (B, H, L, D)
        out = x.new_zeros(B, x.shape[1], L, x.shape[-1]); out[rows, :, cols] = x[0].transpose(0, 1); return out
    o, _ = sdpa_attention_forward(module, pad(query), pad(key), pad(value), None, dropout=dropout, scaling=scaling, is_causal=True)
    return o[rows, cols][None], None  # (B, L, H, D) -> (1, T, H, D)


AttentionInterface.register(NAME, packed_sdpa)
AttentionMaskInterface.register(NAME, lambda *a, **k: None)


BUCKET = 512  # packed length is rounded up to a multiple of this (see packed_hidden)


def packed_hidden(backbone, input_ids, attention_mask):
    m = attention_mask.bool(); lens = m.sum(1)
    assert bool((m == (torch.arange(m.shape[1], device=m.device)[None] < lens[:, None])).all()), 'packing expects right padding'
    rows, cols = m.nonzero(as_tuple=True); n = len(rows)
    # fla's convolution kernels autotune per block count, i.e. per total length (~13 s each on an H200). A trailing
    # filler sequence (its own cu_seqlens segment, so it touches no real row) rounds the length up to a few dozen values.
    fill = -n % BUCKET; ids = input_ids[rows, cols]; pos = cols; seg = lens
    if fill:
        ids = torch.cat([ids, ids.new_zeros(fill)]); pos = torch.cat([pos, torch.arange(fill, device=m.device)])
        seg = torch.cat([lens, lens.new_tensor([fill])])
    cu = torch.zeros(len(seg) + 1, dtype=torch.int32, device=m.device); cu[1:] = seg.cumsum(0)
    mx = int(seg.max())
    h = backbone(input_ids=ids[None], position_ids=pos[None], cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                 max_length_q=mx, max_length_k=mx).last_hidden_state[0, :n]
    return h.new_zeros(*m.shape, h.shape[-1]).index_put((rows, cols), h)


def packed_chunk_forward(self, batch):
    CALLS['packed_forward'] += 1
    h = packed_hidden(self.backbone, batch['input_ids'], batch['attention_mask'])
    positions = batch['positions'].to(h.device); source = h.gather(1, positions.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
    last = h[torch.arange(h.shape[0], device=h.device), batch['last_positions'].to(h.device)]
    mask = batch['document_mask'].unsqueeze(-1).to(device=source.device, dtype=source.dtype); pooled = (source * mask).sum(1) / mask.sum(1).clamp_min(1)
    return {'tokens': self.token_head(source), 'segment': self.segment_head(last), 'mixed': self.mixed_head(last), 'document': self.document_head(pooled)}


def enable(model):
    """Route Detector's physical forwards through the packed path (causal backbones only)."""
    import causal_conv1d
    assert getattr(causal_conv1d, '__version__', '') == 'fla-shim', 'packing needs the fla causal_conv1d shim (sequence boundaries)'
    assert model.kind == 'causal'
    model.backbone.config._attn_implementation = NAME
    # cuDNN SDPA (preferred on H100/H200) builds an execution plan per new (rows, length) shape, ~2.4 s of CPU each;
    # the padded shapes here change every micro-batch. Flash attention has no per-shape setup.
    torch.backends.cuda.enable_cudnn_sdp(False)
    modeling_sweep._chunk_forward = packed_chunk_forward


@torch.no_grad()
def check(model, batch):
    """Hidden states at real tokens, packed vs padded SDPA (call after enable)."""
    cfg = model.backbone.config; cfg._attn_implementation = 'sdpa'
    try:
        ref = model.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state.float()
    finally:
        cfg._attn_implementation = NAME
    c0 = CALLS['attention']; new = packed_hidden(model.backbone, batch['input_ids'], batch['attention_mask']).float()
    m = batch['attention_mask'].bool(); d = (ref - new).abs().amax(-1)
    return {'packed_attention_calls': CALLS['attention'] - c0, 'impl': model.backbone.config._attn_implementation, 'layer_impl': model.backbone.layers[3].self_attn.config._attn_implementation if hasattr(model.backbone.layers[3], 'self_attn') else None,'max_abs': float(d[m].max()), 'mean_abs': float((ref - new).abs()[m].mean()), 'ref_mean_abs': float(ref.abs()[m].mean()),
            'max_abs_first4': float(d[:, :4].max()), 'rel': float((ref - new)[m].norm() / ref[m].norm())}
