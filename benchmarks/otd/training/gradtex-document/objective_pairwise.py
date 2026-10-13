"""Pair-ranking auxiliary loss; no additional model head or fabricated token labels.

The loss value is softplus (not clipped). Its derivative with respect to each
pair score difference is bounded by coefficient * participating-row fraction.
Caller keeps original absolute classification loss and applies this only in
stage2. Both members must be in the SAME microbatch.
"""
import torch
from torch.nn import functional as F


def ranking_loss(token_logits, score_mask, pairs, coefficient=0.1):
    if not 0 <= coefficient <= 0.1:
        raise ValueError('Ranking coefficient outside bounded pilot range')
    if token_logits.ndim != 3 or token_logits.shape[-1] != 2:
        raise ValueError('Expected [batch, tokens, 2] logits')
    if score_mask.shape != token_logits.shape[:2] or score_mask.dtype != torch.bool:
        raise ValueError('Boolean score mask must align with token logits')
    batch_size = token_logits.shape[0]
    used = [int(i) for pair in pairs for i in pair]
    if len(used) != len(set(used)) or any(i < 0 or i >= batch_size for i in used):
        raise ValueError('Every row belongs to at most one complete pair')
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError('Pairs must contain (human row, AI row)')
    if not pairs or coefficient == 0:
        return token_logits.float().sum() * 0
    if not score_mask[used].any(dim=1).all():
        raise ValueError('Every paired row needs a nonempty verified scoring span')
    z = token_logits.float()
    mask = score_mask.float()
    scores = ((z[..., 1] - z[..., 0]) * mask).sum(1) / mask.sum(1).clamp_min(1)
    h = torch.tensor([p[0] for p in pairs], device=z.device)
    a = torch.tensor([p[1] for p in pairs], device=z.device)
    fraction = len(used) / batch_size
    return coefficient * fraction * F.softplus(scores[h] - scores[a]).mean()


def pair_indices(examples):
    """Only explicit verified pair IDs participate; never infer pairs by text."""
    groups = {}
    for i, e in enumerate(examples):
        pid = e.get('ranking_pair_id')
        if pid is None:
            continue
        if not e.get('verified_pair_relation') or e.get('ranking_role') not in ('human', 'ai'):
            raise ValueError('Missing audited pair provenance/role')
        role = e['ranking_role']
        slots = groups.setdefault(pid, {})
        if role in slots:
            raise ValueError('Duplicate role in a pair; use unique draw-pair IDs')
        slots[role] = i
    if any(set(v) != {'human', 'ai'} for v in groups.values()):
        raise ValueError('Incomplete pair crosses microbatch boundary')
    return [(v['human'], v['ai']) for v in groups.values()]
