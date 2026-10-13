"""Per-sentence agreement between the slow and fast scoring paths (run on the H200 from final-20261007/; CPU only).
For each model and score file present on both paths: mean / p99 / max |fast - slow| of sentence AI probabilities, and for the
held-out sets the share of sentences that land on a different side of each path's own in-domain threshold. -> agreement.json"""
import json, sys
from pathlib import Path
import numpy as np
PAIRS = {'moe-mixA-full': ('fastcmp/slow-moe-mixA-full', 'stage2-epoch0'), 'moe-mixB-full': ('sweeps/moe-mixB-full/eval', 'stage2-epoch0'),
         'final-moe': ('fastcmp/slow-final-moe', 'stage2-epoch2')}
SETS = ['', 'heldout-writers', 'wide1-heldout', 't21-heldout-v2', 'cross-model-s500', 'fable-humanizer', 'fable-cowrite', 'fable-polish']


def load(p):
    z = np.load(p); keys = sorted((k for k in z.files if k != 'doc'), key=int)
    return np.concatenate([z[k].reshape(-1, 2) for k in keys]) if keys else np.zeros((0, 2))


out = {}
for tag, (sdir, ck) in PAIRS.items():
    fdir = Path(f'sweeps/fast-{tag}/eval'); out[tag] = {}
    for s in SETS:
        stem = f'{s}-{ck}' if s else ck
        a, b = Path(sdir) / f'{stem}-sentences.npz', fdir / f'{stem}-sentences.npz'
        if not (a.exists() and b.exists()): continue
        x, y = load(a), load(b)
        if x.shape != y.shape or not np.array_equal(x[:, 1], y[:, 1]): out[tag][s or 'sweep'] = {'error': f'shape {x.shape} vs {y.shape}'}; continue
        d = np.abs(y[:, 0] - x[:, 0]); r = {'n': int(len(d)), 'mean_abs': float(d.mean()), 'p99_abs': float(np.quantile(d, .99)), 'max_abs': float(d.max())}
        if s:
            ts = json.loads((Path(sdir) / f'{stem}.json').read_text())['threshold_in_domain']; tf = json.loads((fdir / f'{stem}.json').read_text())['threshold_in_domain']
            flip = (x[:, 0] >= ts) != (y[:, 0] >= tf)
            r.update(thr_slow=ts, thr_fast=tf, flips=int(flip.sum()), flip_share=float(flip.mean()))
        out[tag][s or 'sweep'] = r
Path(sys.argv[1] if len(sys.argv) > 1 else 'agreement.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:1500])
