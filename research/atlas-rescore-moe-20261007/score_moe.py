"""Score clean paper text with the final MoE (Qwen3.6 35B-A3B, run final-moe, last checkpoint) for the Paper Atlas.

Same inference rules as research/human-calibration-20261006/classify_clean_papers.py (the Atlas 4B scorer): 510-token
windows, stride 510 (no overlap), Repeat2 layout, BF16, token AI probability = softmax class 1 averaged over the windows
covering a token, document head = token-weighted mean of window logits. The model is modeling_sweep.Detector (as the
run was trained and evaluated with final-20261007/eval_final.py). Output format matches classify_clean_papers.py, so
sentence_scores_space.py (papers mode) and scripts/build_atlas_details.py work unchanged:
  OUT/tokens/tokens-xxxxx.npz      id/offsets (int32), id/probs (float16)
  OUT/data/summary-xxxxx.parquet   id, source_dataset, pdf_sha256, text_sha256, tokens, windows, document_prob, ...
  OUT/provenance.json, OUT/status.json

Work is split into shards of --rows-per-shard papers over the id-sorted rows. Workers on one host share --claims (claim
files; a dead worker's claim is taken over). Hosts that do not share a filesystem split the shards statically with
--shards A:B (half-open range of shard numbers). Run from a directory holding the training code (modeling_sweep.py,
data.py, adapters_short.py, runtime.py); CUDA_VISIBLE_DEVICES picks the GPU(s); --model-parallel splits the model over
all visible GPUs by layers.

Usage: score_moe.py --source DATASET [--source ...] --output DIR --claims DIR --worker NAME --run RUN_DIR --assets BASE
                    [--checkpoint stage2-epoch2-adapters.safetensors] [--batch-size 32] [--merge] [--compile] [--fixed-pad]
                    [--model-parallel] [--shards A:B] [--limit N] [--ids FILE]
"""
import argparse, hashlib, io, json, os, sys, time
from pathlib import Path

SEQ = 1024  # Repeat2 doubles each 510-token window (1020 + boundaries), rounded up


def atomic(path, raw):
    for attempt in range(5):  # the /data bucket mount occasionally returns EIO
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
            tmp.replace(path); return
        except OSError:
            if attempt == 4: raise
            time.sleep(10 * (attempt + 1))


def digest(raw): return hashlib.sha256(raw).hexdigest()


def load_rows(sources, ids_file=None):
    import pyarrow.parquet as pq
    rows = []
    for source in sources:
        files = sorted((source / 'data').glob('train-*.parquet')) if (source / 'data').exists() else sorted(source.glob('*.parquet'))
        for f in files:
            for r in pq.read_table(f, columns=['id', 'text', 'text_sha256', 'pdf_sha256']).to_pylist():
                rows.append({**r, 'source_dataset': source.name})
    if ids_file:
        keep = set(json.loads(Path(ids_file).read_text())); rows = [r for r in rows if r['id'] in keep]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate paper IDs across sources')
    return sorted(rows, key=lambda r: r['id'])


def windows(n, stride=510, width=510): return [0] if n <= width else sorted(set(list(range(0, n - width + 1, stride)) + [n - width]))


def place_layers(model):
    """Split the model over all visible GPUs by whole decoder layers, heads on GPU 0 (as train_sweep.place_layers)."""
    import torch
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    n = torch.cuda.device_count(); cap = {i: f'{int(torch.cuda.get_device_properties(i).total_memory / 2**30 * .85)}GiB' for i in range(n)}
    layer_types = list({m.__class__.__name__ for name, m in model.backbone.named_modules() if name.startswith('layers.') and name.count('.') == 1})
    mem = get_balanced_memory(model, max_memory=cap, no_split_module_classes=layer_types, low_zero=False)
    dm = infer_auto_device_map(model, max_memory=mem, no_split_module_classes=layer_types)
    for h in ('token_head', 'segment_head', 'mixed_head', 'document_head'): dm[h] = 0
    dispatch_model(model, device_map=dm)
    return dm


def load(args):
    import runtime  # noqa: F401  (puts the host's vendor dir first on sys.path)
    # OTD_EXTRA_PATH; PANGRAM_EXTRA_PATH is its old name, still read
    for extra in reversed([x for x in (os.environ.get('OTD_EXTRA_PATH') or os.environ.get('PANGRAM_EXTRA_PATH', '')).split(':') if x]):
        sys.path.insert(0, extra)  # e.g. the Space's peft 0.21.2, matching the H200 the MoE was trained on
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    import modeling_sweep as ms
    from adapters_short import attach_lora
    cfg = json.loads((args.run / 'run.json').read_text())['config']
    model = ms.Detector.load_base({'repo': str(args.assets), 'revision': None}, cfg['kind']); attach_lora(model, cfg)
    w = load_file(str(args.run / args.checkpoint))
    assert set(w) == {n for n, p in model.named_parameters() if p.requires_grad}, 'adapter keys differ from model'
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in w: p.copy_(w[n].to(p.dtype))
    tok = AutoTokenizer.from_pretrained(str(args.assets), local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = model.to(dtype=torch.bfloat16).eval()
    if args.model_parallel and torch.cuda.device_count() > 1: place_layers(model)
    else: model = model.cuda()
    merged = 0
    if args.merge:
        for m in model.backbone.modules():
            if hasattr(m, 'merge') and hasattr(m, 'lora_A'):
                m.merge(); merged += 1
        assert merged, 'no LoRA modules merged'
    if args.compile: model.backbone = torch.compile(model.backbone, dynamic=False)
    model.physical_microbatch = args.batch_size
    return model, tok, cfg, merged


def score(rows, model, tok, batch_size, fixed_pad):
    import numpy as np, torch
    import modeling_sweep as ms
    from data import layout
    enc = tok([r['text'] for r in rows], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    jobs = []; sums = [np.zeros(len(x), np.float64) for x in enc['input_ids']]; den = [np.zeros(len(x), np.int32) for x in enc['input_ids']]
    doc = [[] for _ in rows]
    for i, ids in enumerate(enc['input_ids']):
        for s in windows(len(ids)) if ids else []: jobs.append((i, s, min(s + 510, len(ids))))
    jobs.sort(key=lambda j: j[2] - j[1])  # length buckets: most windows are full 510 tokens; short tails batch together
    with torch.inference_mode():
        for pos in range(0, len(jobs), batch_size):
            part = jobs[pos:pos + batch_size]; real = len(part)
            if fixed_pad: part = part + [part[-1]] * (batch_size - real)
            ex = []
            for i, a, b in part:
                e = layout(enc['input_ids'][i][a:b], [-100] * (b - a), model.kind, 2, tok.cls_token_id, tok.sep_token_id)
                e.update(source_labels=[-100] * (b - a), target=[False] * (b - a), segment_label=-100, mixed_label=-100, sentence_groups=[],
                         document_label=-100, document_only=False,
                         document_mask=[bool(rows[i]['text'][c:d].strip()) for c, d in enc['offset_mapping'][i][a:b]])
                ex.append(e)
            batch = ms.collate(ex, tok.pad_token_id)
            if fixed_pad and batch['input_ids'].shape[1] < SEQ:
                extra = SEQ - batch['input_ids'].shape[1]
                batch['input_ids'] = torch.nn.functional.pad(batch['input_ids'], (0, extra), value=tok.pad_token_id)
                batch['attention_mask'] = torch.nn.functional.pad(batch['attention_mask'], (0, extra), value=0)
            with torch.autocast('cuda', dtype=torch.bfloat16): out = model(batch)
            probs = out['tokens'].float().softmax(-1)[..., 1].cpu().numpy()
            dl = out['document'].float().cpu().numpy(); wts = batch['document_mask'].sum(1).cpu().numpy()
            for (i, a, b), p, d, w in zip(part[:real], probs, dl, wts):
                sums[i][a:b] += p[:b - a]; den[i][a:b] += 1; doc[i].append((d, float(w)))
    res = []
    for i, r in enumerate(rows):
        n = len(enc['input_ids'][i])
        if n and not (den[i] > 0).all(): raise AssertionError('Uncovered source token')
        p = sums[i] / np.maximum(den[i], 1); off = np.asarray(enc['offset_mapping'][i], dtype=np.int64).reshape(-1, 2)
        w = np.array([x[1] for x in doc[i]]); dlog = np.array([x[0] for x in doc[i]]).reshape(-1, 2)
        sm = lambda z: float(np.exp(z[1] - z.max()) / np.exp(z - z.max()).sum())
        mean = (dlog * w[:, None]).sum(0) / max(w.sum(), 1) if len(w) else None
        res.append((r, off, p, sm(mean) if mean is not None else None, [sm(x) for x in dlog]))
    return res, len(jobs)


def alive(claim):
    try:
        pid, ticks = claim.read_text().split(); stat = Path(f'/proc/{pid}/stat').read_text()
    except (OSError, ValueError): return False
    f = stat.rsplit(')', 1)[1].split(); return f[0] != 'Z' and f[19] == ticks


def claim_shard(claims, k):
    path = claims / f'{k:05}.claim'
    me = f"{os.getpid()} {Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]}"
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, me.encode()); os.close(fd); return path
        except FileExistsError:
            if alive(path): return None
            try: path.rename(path.with_suffix(f'.stale.{os.getpid()}'))
            except FileNotFoundError: pass
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', action='append', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--claims', required=True, type=Path); p.add_argument('--worker', required=True)
    p.add_argument('--run', required=True, type=Path); p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--checkpoint', default='stage2-epoch2-adapters.safetensors')
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--rows-per-shard', type=int, default=100)
    p.add_argument('--merge', action='store_true'); p.add_argument('--compile', action='store_true'); p.add_argument('--fixed-pad', action='store_true')
    p.add_argument('--model-parallel', action='store_true'); p.add_argument('--shards', default='')
    p.add_argument('--order', choices=['asc', 'desc'], default='asc', help='desc: take shards from the end (second host meeting the first in the middle)')
    p.add_argument('--done-dir', action='append', default=[], type=Path, help='other hosts\' synced outputs; shards finished there are skipped')
    p.add_argument('--limit', type=int, default=0); p.add_argument('--ids', default='')
    p.add_argument('--benchmark', action='store_true', help='score --limit papers once, print throughput, write nothing')
    a = p.parse_args()
    import numpy as np, torch
    import pyarrow as pa, pyarrow.parquet as pq
    rows = load_rows(a.source, a.ids or None)
    if a.limit: rows = rows[:a.limit]
    t0 = time.time(); model, tok, cfg, merged = load(a); load_s = time.time() - t0
    if a.benchmark:
        torch.cuda.reset_peak_memory_stats()
        warm = rows[:4]; score(warm, model, tok, a.batch_size, a.fixed_pad)  # warm-up (compile, allocator)
        torch.cuda.synchronize(); t = time.time(); res, nw = score(rows[4:], model, tok, a.batch_size, a.fixed_pad); torch.cuda.synchronize(); dt = time.time() - t
        toks = sum(len(x[2]) for x in res)
        peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
        print(json.dumps({'papers': len(res), 'windows': nw, 'seconds': round(dt, 1), 'papers_per_s': round(len(res) / dt, 3),
                          'windows_per_s': round(nw / dt, 2), 'tokens_per_s': round(toks / dt), 'peak_gb': round(peak, 1), 'load_s': round(load_s),
                          'batch_size': a.batch_size, 'merge': a.merge, 'merged_modules': merged, 'compile': a.compile, 'fixed_pad': a.fixed_pad,
                          'gpus': torch.cuda.device_count(), 'gpu': torch.cuda.get_device_name(0)}), flush=True)
        return
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    manifest = digest('\n'.join(f"{r['id']}\t{r['text_sha256']}" for r in rows).encode())
    prov = {'model': 'qwen36-35b-a3b final-moe', 'checkpoint': a.checkpoint, 'checkpoint_sha256': digest((a.run / a.checkpoint).read_bytes()),
            'run_json_sha256': digest((a.run / 'run.json').read_bytes()), 'sources': [s.name for s in a.source], 'rows': len(rows),
            'paper_manifest_sha256': manifest, 'windows': {'width': 510, 'stride': 510}, 'layout': 'Repeat2', 'precision': 'BF16',
            'rows_per_shard': a.rows_per_shard, 'calibrated': False,
            'outputs': 'token AI probability (class 1) averaged over windows; document head token-weighted mean over windows'}
    if (out / 'provenance.json').exists():
        old = json.loads((out / 'provenance.json').read_text())
        if {k: v for k, v in old.items()} != prov: raise ValueError('Output belongs to another model/input set')
    else: atomic(out / 'provenance.json', json.dumps(prov, indent=2).encode())
    shards = [(k, rows[o:o + a.rows_per_shard]) for k, o in enumerate(range(0, len(rows), a.rows_per_shard))]
    if a.shards:
        lo, hi = map(int, a.shards.split(':')); shards = [s for s in shards if lo <= s[0] < hi]
    claims = a.claims; claims.mkdir(parents=True, exist_ok=True); stop = claims / f'stop-{a.worker}'; stop.unlink(missing_ok=True)
    if a.order == 'desc': shards = shards[::-1]
    def finished():
        got = set()
        for o in [out] + a.done_dir:
            d = set(os.listdir(o / 'data')) if (o / 'data').exists() else set(); t = set(os.listdir(o / 'tokens')) if (o / 'tokens').exists() else set()
            got |= {k for k, _ in shards if f'summary-{k:05}.parquet' in d and f'tokens-{k:05}.npz' in t}
        return got
    def report(state, **x):
        s = {'state': state, 'worker': a.worker, 'pid': os.getpid(), 'updated_at': time.time(), **x}
        atomic(claims / 'workers' / f'{a.worker}.json', json.dumps(s).encode()); print(json.dumps(s), flush=True)
    started = time.time(); mine = 0; papers = 0
    while True:
        if stop.exists(): report('stopped', shards_done=mine); return
        done = finished()
        if len(done) == len(shards): report('complete', shards_done=mine, papers=papers, seconds=round(time.time() - started)); return
        held = None
        for k, batch in shards:
            if k in done: continue
            held = claim_shard(claims, k)
            if held: break
        if held is None: report('waiting', shards_done=mine); time.sleep(30); continue
        if k in finished(): held.unlink(); continue
        report('scoring', shard=k, shards_done=mine, papers=papers, elapsed=round(time.time() - started))
        res, _ = score(batch, model, tok, a.batch_size, a.fixed_pad); table = []; packed = {}
        for r, off, pr, dp, win in res:
            packed[r['id'] + '/offsets'] = off.astype(np.int32); packed[r['id'] + '/probs'] = pr.astype(np.float16)
            table.append({'id': r['id'], 'source_dataset': r['source_dataset'], 'pdf_sha256': r['pdf_sha256'], 'text_sha256': r['text_sha256'],
                          'tokens': len(pr), 'windows': len(win), 'document_prob': dp, 'window_document_prob_max': max(win) if win else None,
                          'token_prob_mean': float(pr.mean()) if len(pr) else None, 'token_prob_max': float(pr.max()) if len(pr) else None,
                          'token_frac_over_0_5': float((pr > 0.5).mean()) if len(pr) else None})
        buf = io.BytesIO(); np.savez_compressed(buf, **packed); atomic(out / 'tokens' / f'tokens-{k:05}.npz', buf.getvalue())
        buf = io.BytesIO(); pq.write_table(pa.Table.from_pylist(table), buf, compression='zstd'); atomic(out / 'data' / f'summary-{k:05}.parquet', buf.getvalue())
        held.unlink(missing_ok=True); mine += 1; papers += len(batch)


if __name__ == '__main__':
    main()
