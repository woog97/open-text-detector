"""Edited rows from the three held-out small-edit sets, prepared for site-style scoring (Calibration page, Oct 12 2026).

The Calibration page's edit curves used 108 GPT-6 Luna edits per size from the evaluation suite. This collects the mixed
rows of heldout-writers, t21-heldout-v2 and wide1-heldout (the sets eval_final.py scores), maps their conditions to the
page's three sizes, and drops every row that shares a 10-word sequence with any training row of either published model
(final-20261007/mix/prepared-v2: stage 1, the merged 72,000-row stage 2, and the selection windows), or whose paper id
prefix appears in that training data.

Usage: build_edit_rows.py EVAL_INPUTS PREPARED OUT_DIR
Writes OUT_DIR/edit-rows.jsonl.gz (rows sections_roc.py pos reads), OUT_DIR/source/data/train-00000.parquet (the
scorers' input) and OUT_DIR/build-stats.json.
"""
import gzip, hashlib, json, re, sys
from collections import Counter
from pathlib import Path

SETS = {'heldout-writers': 'heldout-score-rows.jsonl.gz', 't21-heldout-v2': 't21-heldout-score-rows-v2.jsonl.gz',
        'wide1-heldout': 'wide1-heldout-rows.jsonl.gz'}
SIZE = {'insert_one': 'one', 'rewrite_one': 'one', 'split_one': 'one', 'rewrite_two': 'two',
        'paragraph_draft': 'para', 'paragraph_rewrite': 'para', 'para_draft': 'para', 'para_rewrite': 'para'}
N = 10
words = lambda t: re.findall(r"\w+", t.lower())


def shingles(t):
    w = words(t)
    return {hash(' '.join(w[i:i + N])) for i in range(len(w) - N + 1)}


def paper_key(row_id):
    """16-hex paper prefix from ids like 'arxiv:9c90575ee94172e6/p01/...' or '13e36f06c66134ad.../q001/...'."""
    head = row_id.split('/')[0].split(':')[-1]
    return head[:16] if re.fullmatch(r'[0-9a-f]{16,64}', head) else None


def main(eval_inputs, prepared, out):
    out = Path(out); (out / 'source/data').mkdir(parents=True, exist_ok=True)
    train, train_papers = set(), set()
    for f in ('stage1-epoch0', 'stage2-epoch0', 'selection-windows'):
        for line in gzip.open(Path(prepared) / f'{f}.jsonl.gz', 'rt'):
            r = json.loads(line); train |= shingles(r['text'])
            for k in (paper_key(r['id']), (r.get('paper_id') or '')[:16] or None):
                if k: train_papers.add(k)
    stats = Counter(); rows = []
    for name, fn in SETS.items():
        for line in gzip.open(Path(eval_inputs) / fn, 'rt'):
            r = json.loads(line)
            if r['label'] != 'mixed': continue
            size = SIZE.get(r['condition'])
            if size is None: stats[f'skipped {name} {r["condition"]}'] += 1; continue
            pk = paper_key(r['id']); pid = (r.get('paper_id') or '')[:16]
            if pk in train_papers or (pid and pid in train_papers): stats[f'dropped paper {name}'] += 1; continue
            if shingles(r['text']) & train: stats[f'dropped text {name}'] += 1; continue
            stats[f'kept {name} {size}'] += 1
            rows.append({'id': f'{name}/{r["id"]}', 'text': r['text'], 'regions': r['regions'], 'condition': size,
                         'writer': r['writer'], 'set': name, 'edit': r['condition']})
    with gzip.open(out / 'edit-rows.jsonl.gz', 'wt') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.table({'id': [r['id'] for r in rows], 'text': [r['text'] for r in rows],
                             'text_sha256': [hashlib.sha256(r['text'].encode()).hexdigest() for r in rows],
                             'pdf_sha256': [''] * len(rows)}), out / 'source/data/train-00000.parquet')
    stats = dict(sorted(stats.items())); stats['training_shingles'] = len(train); stats['training_paper_keys'] = len(train_papers)
    (out / 'build-stats.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
