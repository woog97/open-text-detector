"""Atlas scoring through the fast inference path: same papers, windows, shards, claims and output files as score_moe.py.

Only the forward pass differs from score_moe.py. Windows are still 510 tokens, stride 510, the last one shifted to end at
the last token (score_moe.windows), in the Repeat2 layout with no special tokens. They are packed into one sequence of
TOKEN_BUDGET tokens per forward (fast_eval._forward rules: fixed pack length, filler segments), with LoRA merged into
the weights (fast_eval.prepare(merge=True)) and compiled norms. Per window both heads run on the scored copy:
token head per token, document head on the mean hidden state over non-whitespace tokens (as modeling_sweep.Detector),
so the outputs (token probability averaged over windows; document = token-weighted mean of window logits) match.
Needs ccdshim on PYTHONPATH. Usage: as score_moe.py (no --merge/--batch-size/--fixed-pad/--model-parallel), plus
--token-budget.
"""
import argparse, io, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_moe as sm  # load_rows, windows, load, claim_shard, atomic, digest


def load(args):
    args.merge = False; args.model_parallel = False; args.compile = False; args.batch_size = 32
    model, tok, cfg, _ = sm.load(args)
    import fast_eval
    fast_eval.prepare(model, merge=True)
    return model, tok, cfg, len(fast_eval._lora_modules(model))


def forward(model, seqs, sources, masks, total):
    """fast_eval._forward plus the document head. Returns per window (token probs, document logits (2,), weight)."""
    import numpy as np, torch
    dev = next(model.parameters()).device
    lens = [len(s) for s in seqs]; n = sum(lens); assert n <= total
    fills = [1024] * ((total - n) // 1024) + ([(total - n) % 1024] if (total - n) % 1024 else [])
    ids = torch.tensor([t for s in seqs for t in s] + [0] * (total - n), device=dev)
    pos = torch.tensor([i for L in lens + fills for i in range(L)], device=dev)
    seg = lens + fills
    cu = torch.zeros(len(seg) + 1, dtype=torch.int32, device=dev); cu[1:] = torch.tensor(seg, device=dev).cumsum(0)
    mx = max(seg)
    h = model.backbone(input_ids=ids[None], position_ids=pos[None], cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=mx, max_length_k=mx).last_hidden_state[0]
    st = np.cumsum([0] + lens[:-1])
    idx = torch.tensor(np.concatenate([s + np.asarray(src) for s, src in zip(st, sources)]), device=dev)
    src = h[idx]
    p = model.token_head(src).float().softmax(-1)[:, 1].cpu().numpy()
    # Document head: mean of the scored copy's hidden states over the window's non-whitespace tokens.
    owner = torch.tensor(np.repeat(np.arange(len(seqs)), [len(s) for s in sources]), device=dev)
    m = torch.tensor(np.concatenate(masks), device=dev, dtype=src.dtype)
    num = torch.zeros(len(seqs), src.shape[-1], device=dev, dtype=torch.float32).index_add_(0, owner, (src * m[:, None]).float())
    cnt = torch.zeros(len(seqs), device=dev, dtype=torch.float32).index_add_(0, owner, m.float())
    pooled = (num / cnt.clamp_min(1)[:, None]).to(src.dtype)
    d = model.document_head(pooled).float().cpu().numpy(); w = cnt.cpu().numpy()
    cuts = np.cumsum([len(s) for s in sources])[:-1]
    return [(a, b, float(c)) for a, b, c in zip(np.split(p, cuts), d, w)]


def score(rows, model, tok, budget):
    import numpy as np, torch
    from data import layout
    enc = tok([r['text'] for r in rows], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    sums = [np.zeros(len(x), np.float64) for x in enc['input_ids']]; den = [np.zeros(len(x), np.int32) for x in enc['input_ids']]
    doc = [[] for _ in rows]; jobs = []
    for i, ids in enumerate(enc['input_ids']):
        for s in sm.windows(len(ids)) if ids else []: jobs.append((i, s, min(s + 510, len(ids))))
    jobs.sort(key=lambda j: j[2] - j[1])
    batch = []; used = 0

    def flush():
        seqs, srcs, masks = [], [], []
        for i, a, b in batch:
            e = layout(enc['input_ids'][i][a:b], [-100] * (b - a), model.kind, 2, tok.cls_token_id, tok.sep_token_id)
            seqs.append(e['ids']); srcs.append(e['source_positions'])
            masks.append(np.array([bool(rows[i]['text'][c:d].strip()) for c, d in enc['offset_mapping'][i][a:b]]))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = forward(model, seqs, srcs, masks, budget)
        for (i, a, b), (p, d, w) in zip(batch, out):
            sums[i][a:b] += p; den[i][a:b] += 1; doc[i].append((d, w))

    with torch.inference_mode():
        for j in jobs:
            L = 2 * (j[2] - j[1])
            if batch and used + L > budget: flush(); batch = []; used = 0
            batch.append(j); used += L
        if batch: flush()
    res = []
    for i, r in enumerate(rows):
        n = len(enc['input_ids'][i])
        if n and not (den[i] > 0).all(): raise AssertionError('Uncovered source token')
        p = sums[i] / np.maximum(den[i], 1); off = np.asarray(enc['offset_mapping'][i], dtype=np.int64).reshape(-1, 2)
        w = np.array([x[1] for x in doc[i]]); dlog = np.array([x[0] for x in doc[i]]).reshape(-1, 2)
        smx = lambda z: float(np.exp(z[1] - z.max()) / np.exp(z - z.max()).sum())
        mean = (dlog * w[:, None]).sum(0) / max(w.sum(), 1) if len(w) else None
        res.append((r, off, p, smx(mean) if mean is not None else None, [smx(x) for x in dlog]))
    return res, len(jobs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', action='append', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--claims', required=True, type=Path); p.add_argument('--worker', required=True)
    p.add_argument('--run', required=True, type=Path); p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--checkpoint', default='stage2-epoch0-adapters.safetensors'); p.add_argument('--model-name', default='qwen36-35b-a3b final-moe (abl-noseg-nomix)')
    p.add_argument('--token-budget', type=int, default=65536); p.add_argument('--rows-per-shard', type=int, default=100)
    p.add_argument('--shards', default=''); p.add_argument('--order', choices=['asc', 'desc'], default='asc')
    p.add_argument('--limit', type=int, default=0); p.add_argument('--ids', default='')
    p.add_argument('--benchmark', action='store_true', help='score --limit papers once, print throughput, write nothing')
    a = p.parse_args()
    import numpy as np, torch
    import pyarrow as pa, pyarrow.parquet as pq
    rows = sm.load_rows(a.source, a.ids or None)
    if a.limit: rows = rows[:a.limit]
    t0 = time.time(); model, tok, cfg, merged = load(a); load_s = time.time() - t0
    if a.benchmark:
        torch.cuda.reset_peak_memory_stats()
        score(rows[:4], model, tok, a.token_budget)  # warm-up (compiled norms, Triton tuning)
        torch.cuda.synchronize(); t = time.time(); res, nw = score(rows[4:], model, tok, a.token_budget); torch.cuda.synchronize(); dt = time.time() - t
        toks = sum(len(x[2]) for x in res)
        print(json.dumps({'papers': len(res), 'windows': nw, 'seconds': round(dt, 1), 'papers_per_s': round(len(res) / dt, 3),
                          'windows_per_s': round(nw / dt, 2), 'tokens_per_s': round(toks / dt), 'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                          'load_s': round(load_s), 'token_budget': a.token_budget, 'merged_modules': merged, 'gpu': torch.cuda.get_device_name(0)}), flush=True)
        return
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    manifest = sm.digest('\n'.join(f"{r['id']}\t{r['text_sha256']}" for r in rows).encode())
    prov = {'model': a.model_name, 'checkpoint': a.checkpoint, 'checkpoint_sha256': sm.digest((a.run / a.checkpoint).read_bytes()),
            'run_json_sha256': sm.digest((a.run / 'run.json').read_bytes()), 'sources': [s.name for s in a.source], 'rows': len(rows),
            'paper_manifest_sha256': manifest, 'windows': {'width': 510, 'stride': 510}, 'layout': 'Repeat2', 'precision': 'BF16',
            'inference': f'score_atlas_fast.py: packed windows ({a.token_budget} tokens per forward), LoRA merged, compiled norms',
            'rows_per_shard': a.rows_per_shard, 'calibrated': False,
            'outputs': 'token AI probability (class 1) averaged over windows; document head token-weighted mean over windows'}
    if (out / 'provenance.json').exists():
        if json.loads((out / 'provenance.json').read_text()) != prov: raise ValueError('Output belongs to another model/input set')
    else: sm.atomic(out / 'provenance.json', json.dumps(prov, indent=2).encode())
    shards = [(k, rows[o:o + a.rows_per_shard]) for k, o in enumerate(range(0, len(rows), a.rows_per_shard))]
    if a.shards:
        lo, hi = map(int, a.shards.split(':')); shards = [s for s in shards if lo <= s[0] < hi]
    claims = a.claims; claims.mkdir(parents=True, exist_ok=True); stop = claims / f'stop-{a.worker}'; stop.unlink(missing_ok=True)
    if a.order == 'desc': shards = shards[::-1]
    def finished():
        d = set(os.listdir(out / 'data')) if (out / 'data').exists() else set(); t = set(os.listdir(out / 'tokens')) if (out / 'tokens').exists() else set()
        return {k for k, _ in shards if f'summary-{k:05}.parquet' in d and f'tokens-{k:05}.npz' in t}
    def report(state, **x):
        s = {'state': state, 'worker': a.worker, 'pid': os.getpid(), 'updated_at': time.time(), **x}
        sm.atomic(claims / 'workers' / f'{a.worker}.json', json.dumps(s).encode()); print(json.dumps(s), flush=True)
    started = time.time(); mine = 0; papers = 0
    while True:
        if stop.exists(): report('stopped', shards_done=mine); return
        done = finished()
        if len(done) == len(shards): report('complete', shards_done=mine, papers=papers, seconds=round(time.time() - started)); return
        held = None
        for k, batch in shards:
            if k in done: continue
            held = sm.claim_shard(claims, k)
            if held: break
        if held is None: report('waiting', shards_done=mine); time.sleep(30); continue
        if k in finished(): held.unlink(); continue
        report('scoring', shard=k, shards_done=mine, papers=papers, elapsed=round(time.time() - started), total_shards=len(shards), done_shards=len(done))
        res, _ = score(batch, model, tok, a.token_budget); table = []; packed = {}
        for r, off, pr, dp, win in res:
            packed[r['id'] + '/offsets'] = off.astype(np.int32); packed[r['id'] + '/probs'] = pr.astype(np.float16)
            table.append({'id': r['id'], 'source_dataset': r['source_dataset'], 'pdf_sha256': r['pdf_sha256'], 'text_sha256': r['text_sha256'],
                          'tokens': len(pr), 'windows': len(win), 'document_prob': dp, 'window_document_prob_max': max(win) if win else None,
                          'token_prob_mean': float(pr.mean()) if len(pr) else None, 'token_prob_max': float(pr.max()) if len(pr) else None,
                          'token_frac_over_0_5': float((pr > 0.5).mean()) if len(pr) else None})
        buf = io.BytesIO(); np.savez_compressed(buf, **packed); sm.atomic(out / 'tokens' / f'tokens-{k:05}.npz', buf.getvalue())
        buf = io.BytesIO(); pq.write_table(pa.Table.from_pylist(table), buf, compression='zstd'); sm.atomic(out / 'data' / f'summary-{k:05}.parquet', buf.getvalue())
        held.unlink(missing_ok=True); mine += 1; papers += len(batch)


if __name__ == '__main__':
    main()
