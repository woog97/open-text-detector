"""Summarize backbone evaluation predictions into metrics.json for the results page."""
import gzip, json, math, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = [
    ('modernbert', 'ModernBERT-large', 'encoder · full-length run'),
    ('ettin', 'Ettin 1B', 'encoder · 10% run'),
    ('qwen35-4b', 'Qwen3.5 4B', 'decoder LoRA · 10% run'),
    ('qwen35-9b', 'Qwen3.5 9B', 'decoder LoRA · 10% run'),
    ('gemma4-12b', 'Gemma 4 12B', 'decoder LoRA · 10% run'),
    ('qwen36-35b-a3b', 'Qwen3.6 35B-A3B', 'MoE LoRA · 10% run'),
]
PROFILES = ['workflow', 'comparison', 'assistance', 'manuscripts']
DOC_GRAN = {'document_native_label', 'observed_generated_response', 'character_provenance',
            'historical_human_paragraph', 'historical_human_paragraph_with_context',
            'observed_edit_spans', 'publisher_character_spans'}
LEARNER = {'ellipse', 'liang', 'pelic'}
PAPER_HUMAN = {'human_paper_workflow_matched', 'human_paper_workflow_remaining', 'human_paper_remaining', 'paper_v3_target'}
HUMAN_GROUPS = [
    ('Research-paper paragraphs', lambda r: r['dataset'] in PAPER_HUMAN),
    ('Paper paragraphs + context', lambda r: r['dataset'] == 'human_paper_context'),
    ('English-learner essays', lambda r: r['dataset'] in LEARNER),
    ('Other human text (public sets)', lambda r: r['dataset'] not in PAPER_HUMAN | LEARNER | {'human_paper_context'}),
]


def load(model):
    out = {}
    for p in PROFILES:
        f = ROOT / model / p / 'predictions.jsonl.gz'
        if f.exists():
            out[p] = [json.loads(l) for l in gzip.open(f, 'rt')]
    return out


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(rows, key=lambda r: r['document_flag']):
    rows = [r for r in rows if r['document_flag'] is not None]
    k = sum(bool(key(r)) for r in rows)
    return {'k': k, 'n': len(rows), 'rate': k / len(rows) if rows else None, 'ci': wilson(k, len(rows))}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = s.argsort(kind='mergesort'); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(pos, neg, target):
    neg = np.sort(np.asarray(neg))[::-1]
    k = int(math.floor(target * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf  # flag strictly above the (k+1)-th highest human score
    return {'tpr': float(np.mean(np.asarray(pos) > thr)), 'fpr': float(np.mean(neg > thr)), 'threshold': float(thr)}


FPR_GRID = np.unique(np.r_[0.0, np.logspace(-4, 0, 160), np.linspace(0, 1, 101)])


def roc_curve(pos, neg):
    """TPR at each FPR_GRID point (best TPR with FPR <= grid value); exact step ROC sampled for plotting."""
    if len(pos) < 20 or len(neg) < 20:
        return None
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-s, kind='mergesort'); s, y = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # end index of each tied-score group
    tpr = np.r_[0, np.cumsum(y)[last] / len(pos)]; fpr = np.r_[0, np.cumsum(1 - y)[last] / len(neg)]
    idx = np.searchsorted(fpr, FPR_GRID, side='right') - 1
    return {'n_ai': len(pos), 'n_human': len(neg), 'tpr': [round(float(v), 5) for v in tpr[idx]]}


def token_metrics(rows):
    c = np.zeros(4, dtype=np.int64)
    for r in rows:
        if r['token_counts'] is not None and not r.get('span_annotation_conflict'):
            c += np.asarray(r['token_counts'])
    tp, fp, fn, tn = map(int, c)
    prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'precision': prec, 'recall': rec, 'f1': f1, 'human_token_fpr': fp / (fp + tn) if fp + tn else None, 'docs': len(rows)}


def clean(rows, label):
    return [r for r in rows if r['native_label'] == label and r['granularity'] in DOC_GRAN]


def summarize(model):
    d = load(model)
    if len(d) < 4:
        return None
    comp, wf, man, asst = d['comparison'], d['workflow'], d['manuscripts'], d['assistance']
    humans = clean(comp, 'human') + clean(wf, 'human')
    s = {}
    s['human_fpr'] = {name: rate([r for r in humans if f(r)]) for name, f in HUMAN_GROUPS}
    s['human_fpr']['All human documents'] = rate(humans)
    s['human_fpr_by_dataset'] = {ds: rate([r for r in humans if r['dataset'] == ds]) for ds in sorted({r['dataset'] for r in humans})}

    ai_comp = clean(comp, 'ai')
    s['ai_tpr'] = {
        'Public AI text (comparison)': rate(ai_comp),
        'Full generated manuscripts': rate(man),
        'Fully rewritten paper paragraphs': rate(clean(wf, 'ai')),
    }
    s['ai_tpr_by_dataset'] = {ds: rate([r for r in ai_comp if r['dataset'] == ds]) for ds in sorted({r['dataset'] for r in ai_comp})}
    s['manuscripts_by_generator'] = {g: rate([r for r in man if r['generator'] == g]) for g in sorted({r['generator'] for r in man})}

    hc = [r['mean_ai_probability'] for r in clean(comp, 'human')]
    ac = [r['mean_ai_probability'] for r in ai_comp]
    s['auroc'] = {'Comparison (pooled)': auroc(ac, hc)}
    for ds in sorted({r['dataset'] for r in comp}):
        a = [r['mean_ai_probability'] for r in ai_comp if r['dataset'] == ds]
        h = [r['mean_ai_probability'] for r in clean(comp, 'human') if r['dataset'] == ds]
        if len(a) >= 20 and len(h) >= 20:
            s['auroc'][ds] = auroc(a, h)
    hp = [r['mean_ai_probability'] for r in humans if r['dataset'] in PAPER_HUMAN]
    ap = [r['mean_ai_probability'] for r in clean(wf, 'ai')] + [r['mean_ai_probability'] for r in man]
    s['auroc']['Papers: human vs rewritten/generated'] = auroc(ap, hp)
    s['roc'] = {'Comparison (pooled)': roc_curve(ac, hc), 'Papers: human vs rewritten/generated': roc_curve(ap, hp)}
    for ds in sorted({r['dataset'] for r in comp}):
        a = [r['mean_ai_probability'] for r in ai_comp if r['dataset'] == ds]
        h = [r['mean_ai_probability'] for r in clean(comp, 'human') if r['dataset'] == ds]
        c = roc_curve(a, h)
        if c:
            s['roc'][ds] = c
    s['oracle_tpr'] ={f'{int(t*100)}% FPR': tpr_at_fpr(ac, hc, t) for t in (0.01, 0.05)}

    mixed_wf = [r for r in wf if r['native_label'] == 'mixed']
    hit = lambda r: r['sentence_counts'] is not None and r['sentence_counts'][0] > 0
    s['sparse_edits'] = {c: {'detected': rate([r for r in mixed_wf if r['condition'] == c], hit),
                             'tokens': token_metrics([r for r in mixed_wf if r['condition'] == c])}
                         for c in ['sentence', 'two_sentence', 'paragraph_concise', 'paragraph_v3']}
    s['sparse_edits_untouched'] = rate([r for r in wf if r['condition'] == 'untouched'],
                                       lambda r: r['sentence_counts'] is not None and r['sentence_counts'][1] > 0)
    s['spans'] = {
        'Paper edits (paper_v3_target)': token_metrics([r for r in comp if r['dataset'] == 'paper_v3_target' and r['native_label'] == 'mixed']),
        'Workflow reconstructions': token_metrics(mixed_wf),
        'Publisher spans (opai)': token_metrics([r for r in comp if r['dataset'] == 'opai' and r['native_label'] == 'mixed']),
        'Synthetic mixed (local_mixed)': token_metrics([r for r in comp if r['dataset'] == 'local_mixed']),
    }
    s['assistance'] = {c: {'doc_flag': rate([r for r in asst if r['condition'] == c]),
                           'mean_flagged_tokens': float(np.mean([r['flagged_token_fraction'] for r in asst if r['condition'] == c]))}
                       for c in ['proofread', 'light_polish', 'substantial_rewrite']}
    run = json.loads((ROOT / model / 'run.json').read_text())
    s['checkpoint'] = [k for k in run.get('model_files', {}) if k.endswith('.safetensors')]
    return s


if __name__ == '__main__':
    out = {'models': [], 'fpr_grid': [round(float(v), 6) for v in FPR_GRID]}
    for key, name, kind in MODELS:
        s = summarize(key)
        out['models'].append({'key': key, 'name': name, 'kind': kind, 'complete': s is not None, 'metrics': s})
        print(key, 'ok' if s else 'missing')
    (ROOT / 'metrics.json').write_text(json.dumps(out, indent=1))
