"""Sentence-level ROC curves from saved per-token scores (scores/<model>/<profile>/*.npz).

Sentence score and label follow the suite scorer: mean probability of non-whitespace tokens
overlapping the sentence; label kept only when all those tokens share one gold label.
Edit passages use the context view only, so each edited sentence is counted once.
"""
import gzip, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / 'space-suite-v1' / 'package'
sys.path.insert(0, str(PKG))
from common import labels_from_regions  # noqa: E402
from analyze import FPR_GRID, MODELS  # noqa: E402

PROFILES = ['workflow', 'comparison']
EDIT = 'paper_workflow_reconstruction'
HUMAN_PAPER = {'human_paper_workflow_matched', 'human_paper_workflow_remaining'}


def sentences(text):
    return [(m.start(), m.end()) for m in re.finditer(r'\S.*?(?:[.!?](?=\s|$)|$)', text, re.S) if m.group().strip()]


def suite_rows():
    rows = {}
    for p in PROFILES:
        for line in gzip.open(PKG / f'{p}.jsonl.gz', 'rt'):
            r = json.loads(line)
            rows[r['id']] = r
    return rows


def sentence_scores(r, off, prob):
    text = r['text']; off = np.asarray(off)
    valid = np.array([bool(text[a:b].strip()) for a, b in off])
    ys = np.asarray(labels_from_regions(text, off, r['regions']))
    out = []
    for a, b in sentences(text):
        use = valid & (off[:, 1] > a) & (off[:, 0] < b)
        if not use.any():
            continue
        labs = set(ys[use].tolist())
        if len(labs) == 1 and next(iter(labs)) in (0, 1):
            out.append((float(prob[use].mean()), next(iter(labs))))
    return out


def sets_for(r):
    """Which curves a row's positive (1) and negative (0) sentences feed."""
    d, cond, view = r['dataset'], r.get('condition'), r.get('view')
    pos, neg = [], []
    if d == EDIT and r['label'] == 'mixed' and view == 'context':
        size = {'sentence': 'one', 'two_sentence': 'two', 'paragraph_concise': 'para', 'paragraph_v3': 'para'}[cond]
        pos = ['papers_all', f'papers_{size}'] + (['papers_small'] if size in ('one', 'two') else [])
        neg = ['papers_neg']
    elif d in HUMAN_PAPER:
        neg = ['papers_neg']
    elif d in ('paper_v3_target', 'opai', 'paper_pilots_exploratory'):
        pos = neg = [d]
    return pos, neg


def roc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-s, kind='mergesort'); s, y = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0, np.cumsum(y)[last] / len(pos)]; fpr = np.r_[0, np.cumsum(1 - y)[last] / len(neg)]
    idx = np.searchsorted(fpr, FPR_GRID, side='right') - 1
    auc = float(np.trapezoid(tpr, fpr))
    negs = np.sort(neg)[::-1]; k = int(np.floor(0.01 * len(negs)))
    thr = negs[k]
    return {'n_ai': int(len(pos)), 'n_human': int(len(neg)), 'tpr': [round(float(v), 5) for v in tpr[idx]], 'auc': auc,
            'tpr_at_1pct': float(np.mean(pos > thr)),
            'op_05': {'fpr': float(np.mean(neg >= 0.5)), 'tpr': float(np.mean(pos >= 0.5))}}


def model_curves(key, rows):
    base = ROOT / 'scores' / key
    pools = {}
    seen = 0
    for p in PROFILES:
        for f in sorted((base / p).glob('scores-*.npz')):
            z = np.load(f, allow_pickle=False)
            i = 0
            while f'id_{i}' in z.files:
                r = rows[str(z[f'id_{i}'])]
                pos_sets, neg_sets = sets_for(r)
                if pos_sets or neg_sets:
                    for sc, lab in sentence_scores(r, z[f'offsets_{i}'], z[f'probabilities_{i}']):
                        for name in (pos_sets if lab == 1 else neg_sets):
                            pools.setdefault(name, ([], []))[0 if lab == 1 else 1].append(sc)
                seen += 1; i += 1
    neg = pools.pop('papers_neg')[1]
    out = {}
    for name, (pos, own_neg) in pools.items():
        n = neg if name.startswith('papers_') else own_neg
        if len(pos) >= 20 and len(n) >= 20:
            out[name] = roc(np.asarray(pos), np.asarray(n))
    return seen, out


if __name__ == '__main__':
    rows = suite_rows()
    result = {}
    for key, name, _ in MODELS:
        if not (ROOT / 'scores' / key).exists():
            print(key, 'no scores'); continue
        seen, curves = model_curves(key, rows)
        result[key] = curves
        print(key, seen, 'rows', {k: (v['n_ai'], v['n_human'], round(v['auc'], 3), round(v['tpr_at_1pct'], 3)) for k, v in curves.items()})
    (ROOT / 'sentence_roc.json').write_text(json.dumps(result))
