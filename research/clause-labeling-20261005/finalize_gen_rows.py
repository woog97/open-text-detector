"""Merge generated soft rows and add exact binary truth labels for training.

For generated pairs the construction truth is exact, so each row gets `truth_binary_regions`:
AI (1) for assisted or generated characters, except light polish, which counts as human (0) under the
report's rule, the same convention as the soft labels. `regions` stays the soft binary labels, so a training
builder can choose either. Rows flagged never_train are dropped here (generated data has no reason to keep them).

Usage: finalize_gen_rows.py OUT.jsonl.gz SOFT_ROWS.jsonl.gz [more ...]
"""
import gzip, json, sys
from collections import Counter

out_path, srcs = sys.argv[1], sys.argv[2:]
c = Counter()
with gzip.open(out_path, 'wt') as out:
    for f in srcs:
        for r in map(json.loads, gzip.open(f, 'rt')):
            if r.get('never_train'):
                c['dropped_never_train'] += 1; continue
            spans = []
            for t in r['truth_regions']:
                v = 0 if (t['label'] == 0 or t.get('op') == 'polish') else 1
                if spans and spans[-1]['label'] == v and spans[-1]['end'] == t['start']:
                    spans[-1]['end'] = t['end']
                else:
                    spans.append({'start': t['start'], 'end': t['end'], 'label': v})
            r['truth_binary_regions'] = spans
            r['edit_size'] = r.get('edit_type')
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
            c['rows'] += 1; c[f"split:{r['split']}"] += 1; c[f"dataset:{r['dataset']}"] += 1
print(json.dumps(dict(sorted(c.items())), indent=1))
