"""Collect existing source/edited pairs for soft n-gram relabeling of their AI regions.

Sources (all Luna-generated edits of pre-2022 human paper text):
  gap10000  research/data/paper-gap10000-v3-luna-20260930   paragraph rewrites (all splits)
  llm_edit  research/llm-sentence-edits-20261005/llm-edits-v1  one/two-sentence rewrites, inserts, splits
  splice    research/splice-edits-20261005/splices-v1          rewritten sentences spliced into human context
Each output row: {id, dataset, split, never_train, T (edited doc), S (source doc),
  region [a, b] (AI span in T), t_par / s_par [a, b] (the paragraph(s) around the region in T and S)}.
Rows whose stored AI text does not match the region are counted and skipped, never repaired.

Usage: build_relabel_inputs.py OUT.jsonl [--limit-per-dataset N] [--seed S]
"""
import argparse, ast, gzip, json, random, sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'llm-sentence-edits-20261005'))
from build_llm_edits import never_train_hit


def lit(v):
    return v if isinstance(v, list) else ast.literal_eval(v)


def par_bounds(text, a, b):
    """Expand [a, b) to whole blank-line-separated paragraphs."""
    lo = text.rfind('\n\n', 0, a); lo = 0 if lo < 0 else lo + 2
    hi = text.find('\n\n', b); hi = len(text) if hi < 0 else hi
    return [lo, hi]


def from_gap10000(stats):
    by = {}
    for line in open(R / 'data/paper-gap10000-v3-luna-20260930/dataset.jsonl'):
        r = json.loads(line)
        by.setdefault(r['passage_id'], {})[r['operation']] = r
    for pid, v in sorted(by.items()):
        if {'human_original', 'paragraph_generate'} - v.keys():
            stats['gap10000:incomplete'] += 1; continue
        h, g = v['human_original'], v['paragraph_generate']
        reg = next(x for x in lit(g['regions']) if x['label'] == 'ai_rewritten')
        T, S = g['text'], h['text']
        if T[:reg['start']] != S[:reg['source_start']] or T[reg['end']:] != S[reg['source_end']:]:
            stats['gap10000:context_mismatch'] += 1; continue
        yield {'id': g['id'], 'dataset': 'gap10000', 'split': g['split'], 'paper_id': g['paper_id'], 'T': T, 'S': S,
               'region': [reg['start'], reg['end']], 't_par': par_bounds(T, reg['start'], reg['end']),
               's_par': par_bounds(S, reg['source_start'], reg['source_end'])}


def from_edits(path, name, stats):
    for r in map(json.loads, gzip.open(path, 'rt')):
        ai = [x for x in lit(r['regions']) if x['label'] == 1]
        if len(ai) != 1:
            stats[f'{name}:ai_regions!=1'] += 1; continue
        a, b = ai[0]['start'], ai[0]['end']
        T = r['text']; rep = r.get('replaced_human') or ''
        if T[a:b].strip() != (r['inserted_ai'] or '').strip():
            stats[f'{name}:ai_text_mismatch'] += 1; continue
        S = T[:a] + rep + T[b:]
        yield {'id': r['id'], 'dataset': name, 'split': r.get('split', 'train'), 'paper_id': r['paper_id'], 'T': T, 'S': S,
               'region': [a, b], 't_par': par_bounds(T, a, b), 's_par': par_bounds(S, a, a + len(rep)),
               'edit_type': r.get('edit_type') or f"splice-{r.get('edit_size')}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path); ap.add_argument('--limit-per-dataset', type=int); ap.add_argument('--seed', type=int, default=20261006)
    a = ap.parse_args()
    stats = Counter(); rows = []
    for gen in (from_gap10000(stats),
                from_edits(R / 'llm-sentence-edits-20261005/llm-edits-v1.jsonl.gz', 'llm_edit', stats),
                from_edits(R / 'splice-edits-20261005/splices-v1.jsonl.gz', 'splice', stats)):
        got = list(gen)
        if a.limit_per_dataset:
            random.Random(a.seed).shuffle(got); got = got[:a.limit_per_dataset]
        rows += got
    with open(a.out, 'w') as f:
        for r in rows:
            r['never_train'] = bool(never_train_hit({'paper_id': r['paper_id'], 'text': r['T']}))
            stats[f"{r['dataset']}:ok"] += 1; stats[f"{r['dataset']}:never_train"] += r['never_train']
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == '__main__':
    main()
