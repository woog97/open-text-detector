"""Unmerged LoRA for fused MoE experts (peft target_parameters).

peft's ParamWrapper adds scaling * B_e @ A_e to every expert's full weight on every forward (and again in the
recompute and backward), although each token uses only top_k experts. This computes, per routed token,
    y = x @ W_e^T + scaling * (x @ A_e^T) @ B_e^T
with three grouped matmuls over the same expert offsets. Same parameters, names and shapes as peft (adapter files
are unchanged); only the order of rounding differs (the merged path rounds W_e + dW_e to bf16 first).
"""
import os
import types
import torch
from peft.tuners.lora.layer import ParamWrapper


_TRITON = None  # FAST_MOE_GMM=torch|triton|auto (auto: Triton below sm90, where torch's grouped_mm loops over groups)


def _gmm(x, w, offs):
    """(S, in) @ grouped (E, in, out) -> (S, out)."""
    global _TRITON
    if _TRITON is None:
        mode = os.environ.get('FAST_MOE_GMM', 'auto')
        _TRITON = mode == 'triton' or (mode == 'auto' and torch.cuda.get_device_capability(x.device)[0] < 9)
    if _TRITON:
        from gmm_triton import gmm
        return gmm(x.to(w.dtype), w, offs)
    f = getattr(torch.nn.functional, 'grouped_mm', None) or torch._grouped_mm
    return f(x.to(w.dtype), w, offs=offs)


def _lora(x, W, wrapper, offs, E):
    """scaling * x @ dW_e^T for each routed row, from peft's factors without forming dW_e.

    W is (E, out, in). peft's factor layout depends on its version: 0.21 swaps in/out for (E, out, in) weights
    (A: (E*r, in), B: (out, E*r), dW_e = B_e A_e); 0.18 does not (A: (E*r, out), B: (in, E*r), dW_e = (B_e A_e)^T with
    B_e (in, r)). The shapes tell them apart because out != in for both expert projections.
    """
    name = wrapper.active_adapters[0]; r = wrapper.r[name]; s = wrapper.scaling[name]
    A = wrapper.lora_A[name].weight; B = wrapper.lora_B[name].weight; out, inn = W.shape[1], W.shape[2]
    assert out != inn and A.shape[0] == r * E and B.shape[1] == r * E, ('fast_moe: unexpected LoRA factor shapes', tuple(A.shape), tuple(B.shape), tuple(W.shape), r)
    if A.shape[1] == inn and B.shape[0] == out:  # x @ A_e^T @ B_e^T
        A_ = A.reshape(E, r, inn).transpose(-2, -1)  # (E, in, r)
        B_ = B.reshape(out, r, E).permute(2, 0, 1).contiguous().transpose(-2, -1)  # (E, r, out)
    else:
        assert A.shape[1] == out and B.shape[0] == inn, ('fast_moe: unknown LoRA layout', tuple(A.shape), tuple(B.shape), tuple(W.shape))
        A_ = B.reshape(inn, r, E).permute(2, 0, 1).contiguous()  # (E, in, r): x @ B_e
        B_ = A.reshape(E, r, out)  # (E, r, out): then @ A_e
    return s * _gmm(_gmm(x, A_, offs), B_, offs)


def _proj(x, W, wrapper, offs, E):
    return _gmm(x, W.transpose(-2, -1), offs) + _lora(x, W, wrapper, offs, E)


@torch.compiler.disable  # grouped_mm needs the exact operand layouts built here; inductor may re-stride them
def experts_forward(experts, wrappers, hidden_states, top_k_index, top_k_weights):
    E = experts.num_experts; k = top_k_index.size(-1); T = hidden_states.size(0); H = hidden_states.size(-1)
    ids = top_k_index.reshape(-1); ids_g, perm = torch.sort(ids)
    x = hidden_states[perm // k]; w_g = top_k_weights.reshape(-1)[perm]
    offs = torch.cumsum(torch.histc(ids_g.int(), bins=E, min=0, max=E - 1), 0, dtype=torch.int32)
    h = _proj(x, experts.gate_up_proj, wrappers['gate_up_proj'], offs, E)
    h = experts._apply_gate(h)
    h = _proj(h, experts.down_proj, wrappers['down_proj'], offs, E)
    h = h * w_g.unsqueeze(-1)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return h[inv].view(T, k, H).sum(dim=1).to(hidden_states.dtype)


def patch(model):
    """Route every peft-wrapped fused experts module through experts_forward. Returns the number patched."""
    n = 0
    for mod in model.modules():
        if not isinstance(mod, ParamWrapper) or getattr(mod, '_fast_moe', False):
            continue
        chain = [mod]; base = mod.base_layer
        while isinstance(base, ParamWrapper):
            chain.append(base); base = base.base_layer
        wrappers = {w.parameter_name: w for w in chain}
        if set(wrappers) != {'gate_up_proj', 'down_proj'}:
            continue
        for w in chain:
            w._fast_moe = True
            assert len(w.active_adapters) == 1 and not w.merged and not w.disable_adapters
        assert not getattr(base, 'is_transposed', False) and getattr(base, 'has_gate', True) and not getattr(base, 'has_bias', False)
        def fwd(self, x, idx, wts, _b=base, _w=wrappers):
            if any(w.merged for w in _w.values()):  # LoRA already folded into the weights (inference): plain experts
                return _b(x, idx, wts)
            return experts_forward(_b, _w, x, idx, wts)
        mod.forward = types.MethodType(fwd, mod)
        n += 1
    return n


def unit_test(device='cuda', E=8, hidden=64, inter=48, r=8, tokens=512, k=2):
    """peft's own forward vs experts_forward on a tiny fused-experts module, with random (nonzero) LoRA factors."""
    from torch import nn
    from peft import LoraConfig, inject_adapter_in_model

    class Experts(nn.Module):
        def __init__(self):
            super().__init__(); self.num_experts = E; self.act_fn = nn.SiLU()
            self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * inter, hidden) / hidden ** .5)
            self.down_proj = nn.Parameter(torch.randn(E, hidden, inter) / inter ** .5)

        def _apply_gate(self, h):
            g, u = h.chunk(2, dim=-1); return self.act_fn(g) * u

        def forward(self, x, idx, wts):
            out = torch.zeros_like(x)
            for e in range(E):
                t, j = torch.where(idx == e)
                if len(t):
                    h = self._apply_gate(nn.functional.linear(x[t], self.gate_up_proj[e]))
                    out.index_add_(0, t, nn.functional.linear(h, self.down_proj[e]) * wts[t, j, None])
            return out

    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.experts = Experts()
    torch.manual_seed(0)
    m = inject_adapter_in_model(LoraConfig(r=r, lora_alpha=8, target_modules=[], target_parameters=['experts.gate_up_proj', 'experts.down_proj']), Block())
    m = m.to(device=device, dtype=torch.bfloat16)
    for n_, p_ in m.named_parameters():
        if 'lora_' in n_:
            p_.data.normal_(0, .1)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E, device=device)[:k] for _ in range(tokens)]); wts = torch.rand(tokens, k, device=device).softmax(-1).to(torch.bfloat16)
    with torch.no_grad():
        ref = m.experts(x, idx, wts).float()
        assert patch(m) == 1
        new = m.experts(x, idx, wts).float()
    return {'rel': float((ref - new).norm() / ref.norm()), 'max_abs': float((ref - new).abs().max()), 'ref_abs': float(ref.abs().mean())}
