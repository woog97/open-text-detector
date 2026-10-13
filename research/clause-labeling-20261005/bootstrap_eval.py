"""Paired bootstrap over test pairs: macro-F1 differences between splitters at their frozen dev thresholds.

Usage: bootstrap_eval.py PAIRS.jsonl UNITS.jsonl EVAL.json [--conv report] [--n 2000]
"""
import argparse, json
from collections import defaultdict
import numpy as np
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_eval as le

ap = argparse.ArgumentParser()
ap.add_argument('pairs'); ap.add_argument('units'); ap.add_argument('eval'); ap.add_argument('--conv', default='report'); ap.add_argument('--n', type=int, default=2000)
a = ap.parse_args()
pairs = {p['passage_id']: p for p in map(json.loads, open(a.pairs)) if p['state'] == 'ok'}
rows = defaultdict(dict)
for r in map(json.loads, open(a.units)):
    rows[r['passage_id']][r['splitter']] = r
ev = json.load(open(a.eval))
sp = ['luna', 'spacy', 'sentence']
test = [pid for pid in pairs if pairs[pid]['split'] == 'test' and all(s in rows[pid] for s in sp + ['paragraph'])]
conf = {}  # (pid, splitter) -> 3x3 confusion (pred x true), char counts
for pid in test:
    p = pairs[pid]; sg = le.grams(p['source']); le._NORM[id(sg)] = le.norm(p['source'])
    lab = le.char_truth(p, a.conv)[0]; m = le.mask_nonspace(p['target'])
    for s in sp:
        r = rows[pid][s]; th = ev['splitters'][s][a.conv]['thresholds']
        L = [le.lexical(p['target'][x:y], sg) for x, y in r['target_units']]
        pr = le.paint(len(p['target']), r['target_units'], le.predict(r['target_units'], L, r['E'], th))
        c = np.zeros((3, 3))
        np.add.at(c, (pr[m], lab[m]), 1)
        conf[pid, s] = c
def f1(c):
    return np.mean([0 if c[k, k] == 0 else 2 * c[k, k] / (c[k].sum() + c[:, k].sum()) for k in range(3)])
rng = np.random.default_rng(0)
idx = np.arange(len(test))
point = {s: f1(sum(conf[pid, s] for pid in test)) for s in sp}
boots = defaultdict(list)
for _ in range(a.n):
    b = rng.choice(idx, len(idx))
    v = {s: f1(sum(conf[test[i], s] for i in b)) for s in sp}
    for x, y in (('luna', 'spacy'), ('luna', 'sentence'), ('spacy', 'sentence')):
        boots[x, y].append(v[x] - v[y])
print(f'{a.conv} convention, {len(test)} test pairs; macro-F1', {s: round(v, 3) for s, v in point.items()})
for k, v in boots.items():
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f'  {k[0]} - {k[1]}: {point[k[0]] - point[k[1]]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')
