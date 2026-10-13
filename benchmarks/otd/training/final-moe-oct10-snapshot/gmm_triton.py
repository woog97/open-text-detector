"""Grouped matmul in Triton for GPUs where torch's grouped_mm is a per-group loop (sm80: A100).

On an A100, torch._grouped_mm over 256 experts launches one matmul per expert (3.3x the time of a dense matmul with
the same FLOPs at MoE shapes; worse for rank-16 LoRA factors). gmm(x, w, offs) computes, for rows sorted by group,
    out[rows of group e] = x[rows of group e] @ w[e]        x: (S, K), w: (E, K, N) any strides, offs: (E,) int32 ends
in one launch. Backward: dx with the same kernel on w transposed; dw (only when w needs a gradient) with a
segment-reduction kernel. Tiles are found on the GPU (binary search over per-group tile counts), so no host sync.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _find_group(tile_cum, pid, E: tl.constexpr, LOG2E: tl.constexpr):
    # smallest e with tile_cum[e] > pid
    lo = 0; hi = E - 1
    for _ in tl.static_range(LOG2E):
        mid = (lo + hi) // 2
        v = tl.load(tile_cum + mid)
        go_right = v <= pid
        lo = tl.where(go_right, mid + 1, lo); hi = tl.where(go_right, hi, mid)
    return lo


@triton.jit
def _gmm_fwd(X, W, Y, tile_cum, offs, N, K, sxm, sxk, swe, swk, swn, sym, syn,
             E: tl.constexpr, LOG2E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    total = tl.load(tile_cum + E - 1)
    if pid_m >= total:
        return
    e = _find_group(tile_cum, pid_m, E, LOG2E)
    tile0 = tl.where(e > 0, tl.load(tile_cum + e - 1, mask=e > 0, other=0), 0)
    start = tl.where(e > 0, tl.load(offs + e - 1, mask=e > 0, other=0), 0)
    end = tl.load(offs + e)
    rows = start + (pid_m - tile0) * BM + tl.arange(0, BM); mrow = rows < end
    cols = pid_n * BN + tl.arange(0, BN); mcol = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK); mk = ks < K
        a = tl.load(X + rows[:, None] * sxm + ks[None, :] * sxk, mask=mrow[:, None] & mk[None, :], other=0.)
        b = tl.load(W + e.to(tl.int64) * swe + ks[:, None] * swk + cols[None, :] * swn, mask=mk[:, None] & mcol[None, :], other=0.)
        acc = tl.dot(a, b, acc)
    tl.store(Y + rows[:, None] * sym + cols[None, :] * syn, acc.to(Y.dtype.element_ty), mask=mrow[:, None] & mcol[None, :])


@triton.jit
def _gmm_wgrad(X, DY, DW, offs, N, K, sxm, sxk, sym, syn, swe, swk, swn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # DW[e] (K, N) = X[rows of e]^T @ DY[rows of e]
    e = tl.program_id(0); pk = tl.program_id(1); pn = tl.program_id(2)
    start = tl.where(e > 0, tl.load(offs + e - 1, mask=e > 0, other=0), 0); end = tl.load(offs + e)
    ks = pk * BK + tl.arange(0, BK); mk = ks < K
    cols = pn * BN + tl.arange(0, BN); mcol = cols < N
    acc = tl.zeros((BK, BN), dtype=tl.float32)
    for r0 in range(start, end, BM):
        rows = r0 + tl.arange(0, BM); mrow = rows < end
        a = tl.load(X + rows[None, :] * sxm + ks[:, None] * sxk, mask=mk[:, None] & mrow[None, :], other=0.)  # (BK, BM)
        b = tl.load(DY + rows[:, None] * sym + cols[None, :] * syn, mask=mrow[:, None] & mcol[None, :], other=0.)  # (BM, BN)
        acc = tl.dot(a, b, acc)
    tl.store(DW + e.to(tl.int64) * swe + ks[:, None] * swk + cols[None, :] * swn, acc.to(DW.dtype.element_ty), mask=mk[:, None] & mcol[None, :])


def _blk(n, cap):
    return max(16, min(cap, triton.next_power_of_2(n)))


def _fwd(x, w, offs):
    S, K = x.shape; E, K2, N = w.shape
    assert K == K2 and offs.numel() == E and x.dtype == w.dtype
    y = torch.empty(S, N, device=x.device, dtype=x.dtype)
    if S == 0:
        return y
    if N >= 256 and K >= 256:  # expert projections; tuned on A100 at MoE shapes (1.3x a dense matmul of equal FLOPs)
        BM, BN, BK, nw, ns = 128, 256, 32, 8, 4
    else:  # rank-16 LoRA factors
        BM, BN, BK, nw, ns = (64 if K > 64 else 128), _blk(N, 128), _blk(K, 64), 4, 3
    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    tile_cum = torch.cumsum((counts + BM - 1) // BM, 0, dtype=torch.int32)
    grid = (triton.cdiv(S, BM) + E, triton.cdiv(N, BN))
    _gmm_fwd[grid](x, w, y, tile_cum, offs, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2), y.stride(0), y.stride(1),
                   E=E, LOG2E=max(1, (E - 1).bit_length()), BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    return y


def _wgrad(x, dy, offs, E):
    S, K = x.shape; N = dy.shape[1]
    dw = torch.empty(E, K, N, device=x.device, dtype=torch.float32)
    BK = _blk(K, 64); BN = _blk(N, 64)
    _gmm_wgrad[(E, triton.cdiv(K, BK), triton.cdiv(N, BN))](x, dy, dw, offs, N, K, x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
                                                         dw.stride(0), dw.stride(1), dw.stride(2), BM=64, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return dw


class _GMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offs):
        ctx.save_for_backward(x, w, offs)
        return _fwd(x, w, offs)

    @staticmethod
    def backward(ctx, dy):
        x, w, offs = ctx.saved_tensors; dy = dy.contiguous()
        dx = _fwd(dy, w.transpose(1, 2), offs) if ctx.needs_input_grad[0] else None
        dw = _wgrad(x, dy, offs, w.shape[0]).to(w.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw, None


def gmm(x, w, offs):
    """x (S, K) rows sorted by group, w (E, K, N), offs (E,) int32 cumulative row ends -> (S, N)."""
    return _GMM.apply(x, w, offs)


def check(device='cuda', E=64, K=512, N=256, S=4096):
    """Forward and backward vs a per-group loop in fp32."""
    torch.manual_seed(0)
    ids, _ = torch.randint(0, E, (S,), device=device).sort()
    offs = torch.cumsum(torch.bincount(ids, minlength=E), 0).to(torch.int32)
    x = torch.randn(S, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(E, N, K, device=device, dtype=torch.bfloat16) / K ** .5).transpose(1, 2).detach().requires_grad_(True)
    y = gmm(x, w, offs); g = torch.randn_like(y); y.backward(g)
    xr = x.detach().float().requires_grad_(True); wr = w.detach().float().requires_grad_(True); outs = []; o = 0
    for e, c in enumerate(torch.bincount(ids, minlength=E).tolist()):
        outs.append(xr[o:o + c] @ wr[e]); o += c
    yr = torch.cat(outs); yr.backward(g.float())
    rel = lambda a, b: float((a.float() - b).norm() / b.norm())
    return {'y': rel(y, yr), 'dx': rel(x.grad, xr.grad), 'dw': rel(w.grad, wr.grad)}
