"""Fast token scoring for sweep_eval / cross_model_eval: same windows and averaging as sweep_eval.predict.

sweep_eval.predict runs 510-token windows (stride 256, Repeat2 layout for causal backbones) eight at a time, padded,
through every head. predict() here returns the same per-token AI probabilities, but:
  - windows are packed into one sequence per batch, up to TOKEN_BUDGET tokens (no padding compute; packing.py rules),
  - only the token head runs, on the scored copy's positions,
  - the MoE expert LoRA runs on routed tokens only (fast_moe), or all LoRA is merged into the weights (merge=True).
Needs the ccdshim causal_conv1d on sys.path (packing.enable checks).
"""
import numpy as np
import torch
import fast_moe
import packing
from data import layout
import data as _data
from sweep_eval import starts

TOKEN_BUDGET = 65536  # tokens per packed forward (a multiple of packing.BUCKET)


def prepare(model, merge=False, compile_norms=True):
    """Once per loaded model. merge=True folds LoRA into the base weights (do not load other adapters afterwards).

    compile_norms: the RMSNorm / gated RMSNorm modules run as several elementwise passes in eager mode (~15% of a
    forward); torch.compile fuses each into one kernel. Packs have one fixed length, so each compiles once.
    """
    fast_moe.patch(model)  # no-op on the forward once merged (fast_moe checks the merged flag)
    if merge:
        mods = _lora_modules(model); assert mods, 'no LoRA modules'
        # Pristine base weights on the host: switching checkpoints restores them and merges again, instead of
        # subtracting the old delta in bf16 (which would leave rounding residue after every switch).
        model._fe_base = [(p, p.detach().to('cpu', copy=True).pin_memory()) for p in (_base_param(m) for m in mods)]
        for m in mods:
            m.merge()
    packing.enable(model)
    if compile_norms:
        for m in model.backbone.modules():
            if type(m).__name__.endswith(('RMSNorm', 'RMSNormGated')):
                m.compile(dynamic=False)
    return model.eval()


def _lora_modules(model):
    return [m for m in model.modules() if hasattr(m, 'merge') and hasattr(m, 'lora_A')]


def _base_param(m):
    return m.get_param() if hasattr(m, 'parameter_name') else m.get_base_layer().weight


@torch.no_grad()
def set_adapters(model, weights):
    """Load an adapter state dict (names as in model.named_parameters()); re-merge when the model was prepared merged."""
    params = dict(model.named_parameters())
    assert set(weights) == {n for n, p in params.items() if p.requires_grad}, 'adapter names do not match the trainable parameters'
    base = getattr(model, '_fe_base', None)
    if base is not None:
        for p, saved in base:
            p.copy_(saved, non_blocking=True)
        for m in _lora_modules(model):
            m.merged_adapters.clear()
    for n, w in weights.items():
        params[n].copy_(w.to(params[n].dtype))
    if base is not None:
        torch.cuda.synchronize()
        for m in _lora_modules(model):
            m.merge()


def score_checkpoint(model, tok, path, rows):
    """sweep_eval.score_checkpoint through set_adapters and the fast predict (same outputs and post-processing)."""
    import sweep_eval as se
    from safetensors.torch import load_file
    set_adapters(model, load_file(str(path))); preds = predict(rows, model, tok); scored = []
    for r, (off, p) in zip(rows, preds):
        valid = np.array([bool(r['text'][a:b].strip()) for a, b in off])
        scored.append({'slot': r['slot'], 'split': r['split'], 'label': r['label'], 'condition': r.get('condition'),
                       'doc': float(p[valid].mean()) if valid.any() else 0., 'sents': se.sentence_rows(r, off, p)})
    return scored


def _forward(model, seqs, sources, total):
    """seqs: token id lists; sources: per seq, positions of scored tokens. Returns per seq a float32 array of AI probabilities.

    Every pack is filled to exactly `total` tokens with filler segments of at most 1024 tokens (their own cu_seqlens
    entries, so they touch no real window): fla's Triton kernels tune per total length, and a fixed length tunes once.
    """
    dev = next(model.parameters()).device
    lens = [len(s) for s in seqs]; n = sum(lens); assert n <= total
    fills = [1024] * ((total - n) // 1024) + ([(total - n) % 1024] if (total - n) % 1024 else [])
    ids = torch.tensor([t for s in seqs for t in s] + [0] * (total - n), device=dev)
    pos = torch.tensor([i for L in lens + fills for i in range(L)], device=dev)
    seg = lens + fills
    cu = torch.zeros(len(seg) + 1, dtype=torch.int32, device=dev); cu[1:] = torch.tensor(seg, device=dev).cumsum(0)
    mx = max(seg)
    h = model.backbone(input_ids=ids[None], position_ids=pos[None], cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=mx, max_length_k=mx).last_hidden_state[0]
    starts_ = np.cumsum([0] + lens[:-1])
    idx = torch.tensor(np.concatenate([st + np.asarray(src) for st, src in zip(starts_, sources)]), device=dev)
    p = model.token_head(h[idx]).float().softmax(-1)[:, 1].cpu().numpy()
    return np.split(p, np.cumsum([len(s) for s in sources])[:-1])


@torch.inference_mode()
def predict(rows, model, tok, token_budget=None):
    """Drop-in for sweep_eval.predict: [(offsets (n, 2), per-token AI probability (n,))] per row."""
    budget = token_budget or TOKEN_BUDGET
    enc = tok([r['text'] for r in rows], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    sums = [np.zeros(len(x)) for x in enc['input_ids']]; den = [np.zeros(len(x)) for x in enc['input_ids']]; jobs = []
    for i, ids in enumerate(enc['input_ids']):
        for s in starts(len(ids)):
            jobs.append((i, s, min(s + _data.WINDOW, len(ids))))
    jobs.sort(key=lambda j: j[2] - j[1])
    batch = []; used = 0

    def flush():
        seqs, srcs = [], []
        for i, a, b in batch:
            e = layout(enc['input_ids'][i][a:b], [-100] * (b - a), model.kind, 2, tok.cls_token_id, tok.sep_token_id)
            seqs.append(e['ids']); srcs.append(e['source_positions'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = _forward(model, seqs, srcs, budget)
        for (i, a, b), p in zip(batch, out):
            sums[i][a:b] += p; den[i][a:b] += 1
    for j in jobs:
        L = 2 * (j[2] - j[1]) if model.kind == 'causal' else j[2] - j[1] + 2
        if batch and used + L > budget:
            flush(); batch = []; used = 0
        batch.append(j); used += L
    if batch:
        flush()
    return [(np.asarray(off), s / d) for off, s, d in zip(enc['offset_mapping'], sums, den)]
