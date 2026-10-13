"""Compare soft labels with exact construction truth on generated pairs (inside the edited paragraph).

Binary view: truth AI = assisted or generated; polish counted both ways (report rule: polish = human).
Usage: check_soft_vs_truth.py SOFT_ROWS.jsonl.gz
"""
import gzip, json, sys
from collections import Counter, defaultdict

conf = defaultdict(Counter); per_op = defaultdict(Counter); rows = 0
for r in map(json.loads, gzip.open(sys.argv[1], 'rt')):
    if not r.get('truth_regions'):
        continue
    rows += 1; T = r['text']; a0, b0 = r['target_start'], r['target_end']
    soft = [0] * len(T)
    for g in r['soft_regions']:
        soft[g['start']:g['end']] = [g['label']] * (g['end'] - g['start'])
    for t in r['truth_regions']:
        for i in range(max(t['start'], a0), min(t['end'], b0)):
            if T[i].isspace():
                continue
            op = t.get('op', 'context')
            s = soft[i]
            per_op[op][('human', 'assisted', 'generated')[s]] += 1
            for rule in ('report', 'ours'):
                tv = 0 if (t['label'] == 0 or (rule == 'report' and op == 'polish')) else 1
                conf[rule][(tv, int(s > 0))] += 1
print(rows, 'rows')
for rule, c in conf.items():
    tp, fp, fn, tn = c[(1, 1)], c[(0, 1)], c[(1, 0)], c[(0, 0)]
    print(f'{rule:6s} binary char agreement {(tp + tn) / sum(c.values()):.3f}  AI precision {tp / max(1, tp + fp):.3f}  AI recall {tp / max(1, tp + fn):.3f}  human chars flagged AI {fp / max(1, fp + tn):.4f}')
for op, c in sorted(per_op.items()):
    n = sum(c.values()); print(f'  {op:18s} n={n:7d} ' + ' '.join(f'{k}={c[k] / n:.3f}' for k in ('human', 'assisted', 'generated')))
