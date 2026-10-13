"""Make a soft-label copy of a prepared-v2 training directory (A/B arm "soft").

Every stage-1 and stage-2 row whose id has a soft-labeled version (make_soft_rows.py output: gap10000 rewrites,
LLM sentence edits, splices) gets its `regions` replaced by the soft binary regions, mapped into
the row's window (the row text must occur verbatim in the soft row's text). Everything else is copied
unchanged, so the only difference between the arms is the labels. Rows flagged never_train in the soft
file are reported (they should not be in training data at all) but left as they are.

Usage: soften_prepared.py PREPARED_DIR OUT_DIR SOFT_ROWS.jsonl.gz [more SOFT_ROWS ...]
Writes OUT_DIR/{stage2-epoch*.jsonl.gz, manifest.json (updated hashes), soften-stats.json}; other files are copied.
"""
import ast, gzip, hashlib, json, shutil, sys
from collections import Counter
from pathlib import Path


def regions(v):
    v = v if isinstance(v, list) else ast.literal_eval(v)
    return [{'start': int(r['start']), 'end': int(r['end']), 'label': int(r['label'])} for r in v]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    soft = {}
    for f in sys.argv[3:]:
        for r in map(json.loads, gzip.open(f, 'rt')):
            soft[r['source_id']] = r
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    shutil.copytree(src, out)
    man = json.loads((out / 'manifest.json').read_text()); stats = {}
    for f in sorted(out.glob('stage[12]-epoch*.jsonl.gz')):
        c = Counter(); rows = []
        for r in map(json.loads, gzip.open(f, 'rt')):
            s = soft.get(r['id']) or soft.get(r['id'].split('@')[0])  # crop510 appends @offset to cropped rows
            if s is None or not r.get('regions'):
                rows.append(r); c['unchanged'] += 1; continue
            off = s['text'].find(r['text'])
            if off < 0:
                rows.append(r); c['window_not_found'] += 1; continue
            n = len(r['text']); new = []
            for g in s['regions']:
                a, b = max(g['start'] - off, 0), min(g['end'] - off, n)
                if b > a:
                    new.append({'start': a, 'end': b, 'label': g['label']})
            old_ai = sum(g['end'] - g['start'] for g in regions(r['regions']) if g['label'] == 1)
            new_ai = sum(g['end'] - g['start'] for g in new if g['label'] == 1)
            c['softened'] += 1; c['ai_chars_before'] += old_ai; c['ai_chars_after'] += new_ai
            c['never_train_flagged'] += bool(s.get('never_train'))
            rows.append({**r, 'regions': new, 'label_source': 'soft-ngram-v1'})
        b = ''.join(json.dumps(r) + '\n' for r in rows).encode()
        f.write_bytes(gzip.compress(b))
        key = f.name.replace('.jsonl.gz', '')
        if key in man.get('files', {}):
            man['files'][key] = {**man['files'][key], 'sha256': hashlib.sha256(b).hexdigest(), 'rows': len(rows)}
        stats[key] = dict(c)
    man['soft_labels'] = {'from': [str(Path(x).name) for x in sys.argv[3:]], 'stats': stats}
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    (out / 'soften-stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
