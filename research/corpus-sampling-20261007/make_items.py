"""Turn sampled hosts (sample.py hosts output) into writer items: job d gets the V1 edit mix (rewrite one 30%,
rewrite two 30%, insert one 30%, split one 10%), job e alternates draft/rewrite. Writers are assigned round-robin
from --writers (comma list) in the sampled order, so each writer gets the same spread of papers.
  python make_items.py --job d --hosts hosts-sample.jsonl --writers opus,sonnet --out items.jsonl [--heldout-frac 0]
--heldout-frac marks a seeded share of papers split=heldout (evaluation only, e.g. for a new writer)."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'repo/research/llm-sentence-edits-20261005'))
from build_llm_edits import V1_MIX

ap = argparse.ArgumentParser(); ap.add_argument('--job', choices=['d', 'e'], required=True); ap.add_argument('--hosts', required=True)
ap.add_argument('--writers', required=True); ap.add_argument('--out', required=True); ap.add_argument('--heldout-frac', type=float, default=0.)
ap.add_argument('--round', default='wide1')
a = ap.parse_args(); ws = a.writers.split(','); n = {w: 0 for w in ws}; out = []
for i, h in enumerate(map(json.loads, open(a.hosts))):
    w = ws[i % len(ws)]; k = n[w]; n[w] += 1
    held = int(hashlib.sha256(('heldout:' + h['record_id']).encode()).hexdigest(), 16) % 10000 < a.heldout_frac * 10000
    it = {**h, 'writer': w, 'split': 'heldout' if held else 'train', 'cand_id': h['host_id'], 'round': a.round, 'admission': 'wide-pool'}
    if a.job == 'd': it['edit_type'] = V1_MIX[k % len(V1_MIX)]
    else: it['kind'] = 'draft' if k % 2 == 0 else 'rewrite'
    out.append(it)
with open(a.out, 'w') as f:
    for it in out: f.write(json.dumps(it, ensure_ascii=False) + '\n')
from collections import Counter
print(json.dumps({'items': len(out), 'by_writer': dict(Counter(x['writer'] for x in out)), 'heldout': sum(x['split'] == 'heldout' for x in out)}))
