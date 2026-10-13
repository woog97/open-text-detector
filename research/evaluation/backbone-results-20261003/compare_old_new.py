"""Score the Oct 3 backbone runs on the overnight sweep's fixed evaluation set, so old and new results are comparable.

Uses the old runs' saved per-token scores (scores/<model>/{workflow,comparison}/*.npz) and the sweep's metric code
unchanged (pure functions copied from sweep_eval.py at import). Writes comparison.json.
"""
import gzip, json, math, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parents[2] / 'benchmarks' / 'otd' / 'training' / 'overnight-sweep-20261004'
ns = {}
src = (SWEEP / 'sweep_eval.py').read_text()
for name in ['sentences', 'token_labels', 'sentence_rows', 'auroc', 'recall_at', 'metrics']:
    m = re.search(rf'\ndef {name}\(.*?(?=\n\n\n|\n\n@|\nif __name__)', src, re.S)
    exec(m.group(0), {'np': np, 're': re, 'math': math, **ns}, ns)

OLD = {'modernbert': 'ModernBERT-large (full length)', 'qwen35-4b': 'Qwen3.5 4B (10%)', 'qwen35-9b': 'Qwen3.5 9B (10%)',
       'gemma4-12b': 'Gemma 4 12B (10%)', 'qwen36-35b-a3b': 'Qwen3.6 35B-A3B MoE (10%)'}


def score_old(key, rows):
    want = {r['id']: r for r in rows}; got = {}
    for f in sorted((HERE / 'scores' / key).rglob('*.npz')):
        z = np.load(f); i = 0
        while f'id_{i}' in z.files:
            rid = str(z[f'id_{i}'])
            if rid in want:
                got[rid] = (np.asarray(z[f'offsets_{i}']), z[f'probabilities_{i}'])
            i += 1
    assert len(got) == len(rows), (key, len(got), len(rows))
    scored = []
    for r in rows:
        off, p = got[r['id']]; valid = np.array([bool(r['text'][a:b].strip()) for a, b in off])
        scored.append({'slot': r['slot'], 'split': r['split'], 'label': r['label'], 'condition': r.get('condition'),
                       'doc': float(p[valid].mean()), 'sents': ns['sentence_rows'](r, off, p)})
    return {s: ns['metrics'](scored, s) for s in ('dev', 'test', 'all')}


HUMAN_SLOTS = ('edit_context', 'human_untouched', 'human_paper')
SIZES = {'all': ('sentence', 'two_sentence', 'paragraph_concise', 'paragraph_v3'), 'small': ('sentence', 'two_sentence'),
         'one': ('sentence',), 'para': ('paragraph_concise', 'paragraph_v3')}


def honest(rows, sents, docs, fpr=0.01):
    """Cutoffs at `fpr` on the DEV half's human text, applied unchanged to the TEST half."""
    def human(split):
        return np.asarray([s for i, r in enumerate(rows) if r['split'] == split and r['slot'] in HUMAN_SLOTS for s, l in sents[i] if l == 0])
    def v3(split, lab):
        return np.asarray([s for i, r in enumerate(rows) if r['split'] == split and r['slot'] == 'v3' for s, l in sents[i] if l == lab])
    def docsof(split, slots, label=None):
        return np.asarray([docs[i] for i, r in enumerate(rows) if r['split'] == split and r['slot'] in slots and (label is None or r['label'] == label)])
    cut = lambda neg: float(np.sort(neg)[::-1][int(np.floor(fpr * len(neg)))])
    t_s = cut(human('dev')); t_v3 = cut(v3('dev', 0))
    t_doc = cut(docsof('dev', ('human_paper', 'human_untouched'))); t_pub = cut(docsof('dev', ('public',), 'human'))
    out = {'sentence_cutoff': t_s, 'test_human_fpr': float(np.mean(human('test') > t_s))}
    for k, conds in SIZES.items():
        pos = np.asarray([s for i, r in enumerate(rows) if r['split'] == 'test' and r['slot'] == 'edit_context' and r['condition'] in conds for s, l in sents[i] if l == 1])
        out[f'recall_{k}'] = float(np.mean(pos > t_s))
    out['v3_recall'] = float(np.mean(v3('test', 1) > t_v3)); out['v3_test_fpr'] = float(np.mean(v3('test', 0) > t_v3))
    out['standalone_recall'] = float(np.mean(docsof('test', ('ai_standalone',)) > t_doc))
    out['standalone_test_fpr'] = float(np.mean(docsof('test', ('human_paper', 'human_untouched')) > t_doc))
    out['public_recall'] = float(np.mean(docsof('test', ('public',), 'ai') > t_pub)); out['public_test_fpr'] = float(np.mean(docsof('test', ('public',), 'human') > t_pub))
    return out


def scored_old(key, rows):
    want = {r['id']: r for r in rows}; got = {}
    for f in sorted((HERE / 'scores' / key).rglob('*.npz')):
        z = np.load(f); i = 0
        while f'id_{i}' in z.files:
            rid = str(z[f'id_{i}'])
            if rid in want:
                got[rid] = (np.asarray(z[f'offsets_{i}']), z[f'probabilities_{i}'])
            i += 1
    sents, docs = [], []
    for r in rows:
        off, p = got[r['id']]; valid = np.array([bool(r['text'][a:b].strip()) for a, b in off])
        sents.append(ns['sentence_rows'](r, off, p)); docs.append(float(p[valid].mean()))
    return sents, docs


def scored_new(tag, rows):
    z = np.load(SWEEP / 'results' / 'calibration' / tag / 'stage2-epoch2-sentences.npz')
    return [[(float(s), int(l)) for s, l in z[str(i)]] for i in range(len(rows))], [float(x) for x in z['doc']]


if __name__ == '__main__':
    rows = [json.loads(l) for l in gzip.open(SWEEP / 'sweep-eval-rows.jsonl.gz', 'rt')]
    out = {}
    for key, label in OLD.items():
        if (HERE / 'scores' / key).exists():
            out[key] = {'label': label, **score_old(key, rows)}
            t = out[key]['test']
            print(f"{label:32s} all {t['edits_all']['recall_at_1pct']:.3f} small {t['edits_small']['recall_at_1pct']:.3f} "
                  f"para {t['edits_para']['recall_at_1pct']:.3f} v3 {t['v3_sentences']['recall_at_1pct']:.3f} "
                  f"sa {t['standalone_rewrites']['recall_at_1pct']:.3f} pub {t['public_docs']['auroc']:.3f}")
    (HERE / 'comparison-old-on-sweep-set.json').write_text(json.dumps(out, indent=1))
    hon = {}
    for key in OLD:
        if (HERE / 'scores' / key).exists():
            hon['old:' + key] = honest(rows, *scored_old(key, rows))
    for d in sorted((SWEEP / 'results' / 'calibration').iterdir()):
        if (d / 'stage2-epoch2-sentences.npz').exists():
            hon['new:' + d.name] = honest(rows, *scored_new(d.name, rows))
    (HERE / 'honest-1pct.json').write_text(json.dumps(hon, indent=1))
    for k, v in hon.items():
        print(f"{k:24s} testFPR {v['test_human_fpr']:.4f} all {v['recall_all']:.3f} small {v['recall_small']:.3f} one {v['recall_one']:.3f} para {v['recall_para']:.3f} v3 {v['v3_recall']:.3f}/{v['v3_test_fpr']:.3f} sa {v['standalone_recall']:.3f}/{v['standalone_test_fpr']:.3f} pub {v['public_recall']:.3f}/{v['public_test_fpr']:.3f}")
