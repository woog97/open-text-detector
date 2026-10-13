"""Sample sentences for the hand-annotated clause-boundary gold set.

Half come from human source paragraphs, half from the Luna rewrites of those paragraphs
(always from the edited span). Edge cases (citations, math, parentheses, semicolons,
lists, long sentences) are oversampled to about half the set. Rows are shuffled with
human/AI interleaved, so any prefix the annotator finishes is a balanced random sample.

Usage: sample_gold.py DATASET OUT.jsonl [--n 300]
"""
import argparse, ast, json, random, re
from pathlib import Path

TAGS = {
    'citation': re.compile(r'\[\d[\d,\s–-]*\]|\(\w[^()]*\b(19|20)\d\d\w?\)|et al\.'),
    'math': re.compile(r'[$=<>±∑∫^_\\]|\b[a-zA-Z]\(\w\)'),
    'parenthesis': re.compile(r'\([^)]{8,}\)'),
    'semicolon': re.compile(r';'),
    'list': re.compile(r'\((i+|[a-d1-4])\)|\b(i+|[a-d])\)|:\s'),
}


def regions(g):
    return g['regions'] if isinstance(g['regions'], list) else ast.literal_eval(g['regions'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--seed', type=int, default=20261006)
    a = ap.parse_args()
    by = {}
    for line in open(a.dataset):
        r = json.loads(line)
        by.setdefault(r['passage_id'], {})[r['operation']] = r
    pools = {'human': [], 'ai': []}
    for pid, v in sorted(by.items()):
        if not {'human_original', 'paragraph_generate'} <= v.keys():
            continue
        g = v['paragraph_generate']; reg = next(x for x in regions(g) if x['label'] == 'ai_rewritten')
        for side, row, lo, hi in (('human', v['human_original'], reg['source_start'], reg['source_end']),
                                  ('ai', g, reg['start'], reg['end'])):
            for s in row['sentences']:
                t = s['text']
                if s['start'] >= lo and s['end'] <= hi and 40 <= len(t) <= 450:
                    tags = [k for k, rx in TAGS.items() if rx.search(t)] + (['long'] if len(t) > 250 else [])
                    pools[side].append({'paper_id': row['paper_id'], 'passage_id': pid, 'side': side,
                                        'start': s['start'], 'end': s['end'], 'text': t, 'tags': tags})
    rng = random.Random(a.seed)
    picked = {}
    for side, pool in pools.items():
        rng.shuffle(pool)
        seen, hard, easy = set(), [], []
        for s in pool:  # at most one sentence per paper and side, for spread
            if s['paper_id'] in seen:
                continue
            seen.add(s['paper_id']); (hard if s['tags'] else easy).append(s)
        k = a.n // 2
        picked[side] = hard[:k // 2] + easy[:k - k // 2]
        rng.shuffle(picked[side])
    rows = [x for pair in zip(picked['human'], picked['ai']) for x in pair]
    with open(a.out, 'w') as f:
        for i, s in enumerate(rows):
            f.write(json.dumps(dict(gold_id=f'g{i:03d}', **s), ensure_ascii=False) + '\n')
    print(len(rows), 'sentences;', sum(bool(s['tags']) for s in rows), 'with edge-case tags;', len({s['paper_id'] for s in rows}), 'papers')


if __name__ == '__main__':
    main()
