"""Merge per-rank FSDP2 resume files (train_sweep.py --fsdp) into one single-process resume-state.pt.

Usage: merge_fsdp_resume.py [--micro-batch OLD:NEW] [--peft021 E:r] OUT.pt RANK0.pt RANK1.pt [...]      (needs CUDA and bitsandbytes for 8-bit states)
       merge_fsdp_resume.py --selftest

Moves a run from FSDP over N GPUs to one GPU (or plain DDP) with the same effective batch and micro-batch, e.g. the
Oct 8 mixA-full run from 2 Space A100s to 1 H200. FSDP2 shards every parameter along dim 0 (torch.chunk), so the
trainable LoRA/head tensors are concatenated in rank order. The optimizer ran per shard (ShardOptimizer): parameters in
parameter_groups order (backbone, then token/segment/mixed/document heads), skipping a rank's empty shards. Its 8-bit
AdamW states are concatenated directly when every shard but the last fills whole 256-value quantization blocks;
otherwise they are dequantized, concatenated and re-quantized. Python/NumPy/torch RNG state comes from rank 0.
"""
import sys
import torch

BLOCK = 256; MIN8 = 4096; HEADS = ('token_head.', 'segment_head.', 'mixed_head.', 'document_head.')
WRAP = '__bnb_optimizer_quant_state__'


def group_order(names):
    """Parameter names in the optimizer's order: parameter_groups() lists backbone parameters, then the heads in HEADS order."""
    g0 = [n for n in names if n.startswith('backbone.')]
    g1 = [n for h in HEADS for n in names if n.startswith(h)]
    assert len(g0) + len(g1) == len(names), 'trainable parameters outside backbone and heads'
    return [g0, g1]


def unwrap(s):
    s = dict(s); s.update(s.pop(WRAP, {})); return s


def to_fp32(s, key):
    """One Adam moment of one shard as FP32 (dequantized if 8-bit)."""
    v = s[f'state{key}']
    if v.dtype != torch.uint8:
        return v.float()
    import bitsandbytes.functional as F
    out = F.dequantize_blockwise(v.cuda(), absmax=s[f'absmax{key}'].cuda(), code=s[f'qmap{key}'].cuda(), blocksize=BLOCK)
    return out.float().cpu().view(v.shape)


def merge_state(parts):
    """parts: per-rank bnb states of one parameter's shards, rank order. Returns the full parameter's state."""
    n = sum(p['state1'].numel() for p in parts); steps = {int(p['step']) for p in parts}
    assert len(steps) == 1, f'shards at different optimizer steps {steps}'
    out = {'step': parts[0]['step']}
    if 'unorm_vec' in parts[0]:
        raise SystemExit('max_unorm states are not supported')
    if n < MIN8:  # the full parameter gets 32-bit states
        for k in '12':
            out[f'state{k}'] = torch.cat([to_fp32(p, k) for p in parts])
        return out
    aligned = all(p['state1'].dtype == torch.uint8 for p in parts) and all(p['state1'].numel() % BLOCK == 0 for p in parts[:-1])
    for k in '12':
        if aligned:
            out[f'state{k}'] = torch.cat([p[f'state{k}'] for p in parts]); out[f'absmax{k}'] = torch.cat([p[f'absmax{k}'] for p in parts])
            out[f'qmap{k}'] = parts[0][f'qmap{k}']
        else:
            import bitsandbytes.functional as F
            full = torch.cat([to_fp32(p, k) for p in parts]).cuda()
            code = next((p[f'qmap{k}'] for p in parts if f'qmap{k}' in p), None)
            if code is None:
                code = F.create_dynamic_map(signed=(k == '1'))
            q, qs = F.quantize_blockwise(full, code=code.cuda(), blocksize=BLOCK)
            out[f'state{k}'] = q.cpu().view(full.shape); out[f'absmax{k}'] = qs.absmax.cpu(); out[f'qmap{k}'] = code.cpu()
        assert out[f'absmax{k}'].numel() == -(-n // BLOCK)
    return out


def merge(states):
    s0 = states[0]
    names = list(s0['trainable'])
    for s in states[1:]:
        assert list(s['trainable']) == names, 'ranks list different trainable parameters'
        for k in ('global_step', 'stage', 'epoch', 'batch', 'done_rows'):
            assert s[k] == s0[k], f'ranks differ in {k}: {s[k]} vs {s0[k]}'
    trainable = {n: torch.cat([s['trainable'][n] for s in states]) for n in names}
    groups = group_order(names)
    # Each rank's optimizer indices: running count over parameter_groups order, skipping empty shards on that rank.
    per_rank = []
    for s in states:
        idx = {}; i = 0
        for g in groups:
            for n in g:
                if s['trainable'][n].numel():
                    idx[n] = i; i += 1
        saved = s['opt']['param_groups']
        assert [len(g['params']) for g in saved] == [sum(1 for n in g if n in idx) for g in groups], 'optimizer groups do not match the trainable shards'
        per_rank.append(idx)
    new_state = {}; j = 0; stats = {'aligned_or_32bit': 0, 'requantized': 0}
    for g in groups:
        for n in g:
            parts = [unwrap(s['opt']['state'][idx[n]]) for s, idx in zip(states, per_rank) if n in idx and idx[n] in s['opt']['state']]
            if parts:
                assert len(parts) == sum(1 for idx in per_rank if n in idx), f'{n}: optimizer state missing on some ranks'
                m = merge_state(parts)
                assert m['state1'].numel() == trainable[n].numel(), f'{n}: state {m["state1"].numel()} vs parameter {trainable[n].numel()}'
                m['state1'] = m['state1'].view(trainable[n].shape); m['state2'] = m['state2'].view(trainable[n].shape)
                stats['requantized' if 'absmax1' in m and not all(p['state1'].numel() % BLOCK == 0 for p in parts[:-1]) else 'aligned_or_32bit'] += 1
                new_state[j] = m
            j += 1
    pg = [{**{k: v for k, v in sg.items() if k != 'params'}, 'params': list(range(o, o + len(g)))}
          for sg, g, o in zip(s0['opt']['param_groups'], groups, [0, len(groups[0])])]
    out = {**s0, 'trainable': trainable, 'opt': {'state': new_state, 'param_groups': pg}}
    return out, stats


def to_peft021(m, E, r):
    """Expert LoRA factors from peft 0.18 layout (Space) to 0.21 (H200), see fast_moe._lora:
    0.18: A (E*r, out), B (in, E*r); 0.21: A (E*r, in), B (out, E*r); same per-expert update. Adam moments are elementwise,
    so they get the same permutation (dequantized, permuted, re-quantized)."""
    import bitsandbytes.functional as F
    names = list(m['trainable']); j = {n: i for i, n in enumerate(n for g in group_order(names) for n in g)}
    def fa(Bt, inn): return Bt.reshape(inn, r, E).permute(2, 1, 0).reshape(E * r, inn).contiguous()   # new A from old B
    def fb(At, out): return At.reshape(E, r, out).permute(2, 1, 0).reshape(out, E * r).contiguous()   # new B from old A
    def moment(st, k, shape):
        return to_fp32(st, k).view(shape)
    def requant(st_new, k, full):
        if 'absmax1' not in st_new and full.numel() < MIN8:
            st_new[f'state{k}'] = full.float(); return
        code = F.create_dynamic_map(signed=(k == '1'))
        q, qs = F.quantize_blockwise(full.float().cuda().contiguous(), code=code.cuda(), blocksize=BLOCK)
        st_new[f'state{k}'] = q.cpu().view(full.shape); st_new[f'absmax{k}'] = qs.absmax.cpu(); st_new[f'qmap{k}'] = code.cpu()
    done = 0
    for a in [n for n in names if '.mlp.experts.' in n and '.lora_A.' in n]:
        b = a.replace('.lora_A.', '.lora_B.'); A, B = m['trainable'][a], m['trainable'][b]
        out, inn = A.shape[1], B.shape[0]; assert A.shape[0] == E * r and B.shape[1] == E * r, (a, A.shape, B.shape)
        m['trainable'][a], m['trainable'][b] = fa(B, inn), fb(A, out)
        sa, sb = m['opt']['state'][j[a]], m['opt']['state'][j[b]]; na, nb = {'step': sb['step']}, {'step': sa['step']}
        for k in '12':
            requant(na, k, fa(moment(sb, k, B.shape), inn)); requant(nb, k, fb(moment(sa, k, A.shape), out))
        m['opt']['state'][j[a]], m['opt']['state'][j[b]] = na, nb; done += 1
    return done


def selftest_layout(E=4, r=3, inn=5, out=7):
    """0.18 and 0.21 factors after to_peft021's maps give the same per-expert x @ dW_e^T (fast_moe._lora formulas)."""
    A18, B18 = torch.randn(E * r, out), torch.randn(inn, E * r)
    A21 = B18.reshape(inn, r, E).permute(2, 1, 0).reshape(E * r, inn); B21 = A18.reshape(E, r, out).permute(2, 1, 0).reshape(out, E * r)
    x = torch.randn(10, inn)
    for e in range(E):
        y18 = x @ B18.reshape(inn, r, E).permute(2, 0, 1)[e] @ A18.reshape(E, r, out)[e]
        y21 = x @ A21.reshape(E, r, inn).transpose(-2, -1)[e] @ B21.reshape(out, r, E).permute(2, 0, 1).transpose(-2, -1)[e]
        assert torch.allclose(y18, y21, atol=1e-5), e
    print('layout selftest ok')


def selftest():
    """Shard optimizers vs one full optimizer on the same gradients: merged state must match the full one closely."""
    import bitsandbytes as bnb
    torch.manual_seed(0); shapes = {'backbone.a': (300, 64), 'backbone.b': (129, 128), 'backbone.c': (3, 5), 'token_head.w': (1, 7)}
    full = {n: torch.nn.Parameter(torch.randn(s, device='cuda')) for n, s in shapes.items()}
    shards = [{n: torch.nn.Parameter(p.detach().chunk(2)[r].clone()) if len(p.detach().chunk(2)) > r else torch.nn.Parameter(p.detach()[:0].clone())
               for n, p in full.items()} for r in range(2)]
    of = bnb.optim.AdamW8bit([{'params': [full[n] for n in shapes if n.startswith('backbone')]}, {'params': [full['token_head.w']]}])
    osh = [bnb.optim.AdamW8bit([{'params': [s[n] for n in shapes if n.startswith('backbone') and s[n].numel()]},
                                {'params': [s['token_head.w']] if s['token_head.w'].numel() else []}]) for s in shards]
    for _ in range(5):
        for n, p in full.items():
            g = torch.randn_like(p); p.grad = g
            for r, s in enumerate(shards):
                ch = g.chunk(2)
                if s[n].numel(): s[n].grad = ch[r].clone()
        of.step(); [o.step() for o in osh]
    rng = {'python': None}
    states = [{'trainable': {n: s[n].detach().cpu() for n in shapes}, 'opt': o.state_dict(), 'global_step': 5, 'stage': 2, 'epoch': 0,
               'batch': 0, 'done_rows': 0, 'rng': rng} for s, o in zip(shards, osh)]
    m, stats = merge(states)
    for n in shapes:
        assert torch.equal(m['trainable'][n], torch.cat([s[n].detach().cpu() for s in shards])), n
        print(n, 'merged vs full-optimizer weights max diff', f"{(m['trainable'][n] - full[n].detach().cpu()).abs().max().item():.3g}")
    ref = of.state_dict()['state']; names = [n for n in shapes if n.startswith('backbone')] + ['token_head.w']
    for j, n in enumerate(names):
        a, b = unwrap(ref[j]), m['opt']['state'][j]
        parts = [unwrap(o.state_dict()['state'][[x for x in names if s[x].numel()].index(n)]) for o, s in zip(osh, shards) if s[n].numel()]
        for k in '12':
            src = torch.cat([to_fp32(p, k) for p in parts]).view(-1); got = to_fp32(b, k).view(-1)
            d = (src - got).abs().max().item(); s_ = src.abs().max().item()
            print(n, f'state{k}', b[f'state{k}'].dtype, 'merged vs shards max diff', f'{d:.3g}', 'of', f'{s_:.3g}',
                  '| vs full optimizer', f'{(to_fp32(a, k).view(-1) - got).abs().max().item():.3g}')
            assert d <= 0.01 * s_ + 1e-12, n
    o1 = bnb.optim.AdamW8bit([{'params': [full[n] for n in names[:-1]]}, {'params': [full['token_head.w']]}]); o1.load_state_dict(m['opt'])
    print('selftest ok', stats)


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        selftest_layout(); selftest(); sys.exit()
    peft021 = None
    if sys.argv[1] == '--peft021':  # --peft021 E:r  convert expert LoRA factors from peft 0.18 to 0.21 layout
        peft021 = tuple(map(int, sys.argv[2].split(':'))); sys.argv[1:3] = []
    mb_from = mb_to = None
    if sys.argv[1] == '--micro-batch':  # e.g. --micro-batch 16:32: the data position counts micro-batches of the old size
        mb_from, mb_to = map(int, sys.argv[2].split(':')); sys.argv[1:3] = []
    out, ins = sys.argv[1], sys.argv[2:]
    states = [torch.load(f, map_location='cpu', weights_only=False) for f in ins]
    m, stats = merge(states)
    if peft021:
        stats['peft021_pairs'] = to_peft021(m, *peft021)
    if mb_from:
        rows_done = m['batch'] * mb_from
        assert rows_done % mb_to == 0 and 32 % mb_to == 0, f'position {m["batch"]} x {mb_from} rows does not fall on a micro-batch of {mb_to}'
        m['batch'] = rows_done // mb_to
        assert not m.get('skip'), 'skipped-batch list uses old micro-batch indices'
    tmp = out + '.tmp'; torch.save(m, tmp)
    import os; os.replace(tmp, out)
    print({'out': out, 'global_step': m['global_step'], 'stage': m['stage'], 'batch': m['batch'], 'params': len(m['trainable']),
           'optimizer_states': len(m['opt']['state']), **stats})
