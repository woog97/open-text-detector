"""Calibrate the fast10 Qwen3.5-4B deployment config (stride 510, merged LoRA) at 1% FPR.

Inputs (downloaded from the training-storage bucket into CACHE, never committed):
  suite-sentences.json.gz      eval-suite rows with per-sentence (score, gold) pairs
  suite summary parquet        document probabilities for the 120 AI manuscripts
  calibration sentences/summary  215 human papers (year <= 2022, training overlap removed)
Human negatives are split by paper (stable hash) into a threshold half and a check half.
Writes results.json (aggregates only).
"""
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path(sys.argv[1])
sys.path.insert(0, str(HERE.parent / 'backbone-results-20261003'))
from sentence_roc import sets_for  # noqa: E402

FPR_GRID = np.unique(np.r_[np.geomspace(1e-4, 1, 81), np.linspace(0, 1, 101)])


def roc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-s, kind='mergesort'); s, y = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0, np.cumsum(y)[last] / len(pos)]; fpr = np.r_[0, np.cumsum(1 - y)[last] / len(neg)]
    idx = np.searchsorted(fpr, FPR_GRID, side='right') - 1
    return {'fpr': [round(float(v), 6) for v in FPR_GRID], 'tpr': [round(float(v), 5) for v in tpr[idx]],
            'auc': float(np.trapezoid(tpr, fpr)), 'n_ai': int(len(pos)), 'n_human': int(len(neg))}


def threshold_at(neg, fpr=0.01):
    """Smallest threshold whose flag rate (score > t) on these negatives is <= fpr."""
    negs = np.sort(np.asarray(neg, float))[::-1]
    return float(negs[int(np.floor(fpr * len(negs)))])


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    lo = beta.ppf(a / 2, k, n - k + 1) if k else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def half(pid):
    return int(hashlib.sha256(('cal-split-v1:' + pid).encode()).hexdigest(), 16) % 2


def main():
    import pyarrow.parquet as pq
    suite = json.load(gzip.open(CACHE / 'suite-sentences.json.gz', 'rt'))
    cal_sent = np.load(CACHE / 'cal-sentences.npz')
    cal = {}
    for k in cal_sent.files:
        pid, kind = k.rsplit('/', 1)
        if kind == 'scores': cal[pid] = cal_sent[k].astype(float)
    cal_docs = pq.read_table(CACHE / 'cal-summary.parquet').to_pandas().set_index('id')
    suite_docs = pq.read_table(CACHE / 'suite-summary.parquet').to_pandas().set_index('id')

    # Sentence level
    pos = {k: [] for k in ['papers_all', 'papers_one', 'papers_two', 'papers_para', 'papers_small']}
    suite_neg = []
    for r in suite:
        if 'sentences' not in r: continue
        p_sets, n_sets = sets_for(r)
        for score, gold in r['sentences']:
            if gold == 1:
                for k in p_sets:
                    if k in pos: pos[k].append(score)
            elif gold == 0 and 'papers_neg' in n_sets:
                suite_neg.append(score)
    split = {pid: half(pid) for pid in cal}
    neg_a = np.concatenate([cal[p] for p in cal if split[p] == 0])
    neg_b = np.concatenate([cal[p] for p in cal if split[p] == 1])
    neg_all = np.concatenate(list(cal.values()))
    t_sent = threshold_at(neg_a)
    t_sent_all = threshold_at(neg_all)
    k_b = int((neg_b > t_sent).sum())
    # Per-paper flag rate among held-out human papers: how unevenly the 1% falls
    per_paper_b = [float((cal[p] > t_sent).mean()) for p in cal if split[p] == 1 and len(cal[p])]
    sentence = {
        'threshold_half_a': t_sent, 'threshold_all_human': t_sent_all,
        'heldout_fpr': {'flagged': k_b, 'n': int(len(neg_b)), 'rate': k_b / len(neg_b), 'ci95': clopper_pearson(k_b, len(neg_b))},
        'heldout_papers': {'n': len(per_paper_b), 'flag_rate_p50': float(np.median(per_paper_b)),
                           'flag_rate_p90': float(np.quantile(per_paper_b, .9)), 'flag_rate_max': float(max(per_paper_b)),
                           'papers_with_any_flag': float(np.mean(np.array(per_paper_b) > 0))},
        'roc_vs_human_papers': {k: roc(v, neg_all) for k, v in pos.items()},
        'roc_vs_suite_human_passages': {k: roc(v, suite_neg) for k, v in pos.items()},
        'tpr_at_threshold': {k: float(np.mean(np.asarray(v) > t_sent_all)) for k, v in pos.items()},
        'n_human_sentences': int(len(neg_all)), 'n_human_papers': len(cal),
    }

    # Document level
    ai_docs = [r['id'] for r in suite if r['profile'] == 'manuscripts']
    doc = {}
    for col in ['document_prob', 'token_prob_mean']:
        p = suite_docs.loc[ai_docs, col].to_numpy(float); n = cal_docs[col].to_numpy(float)
        t = threshold_at(n)
        doc[col] = {**roc(p, n), 'threshold_1pct': t, 'tpr_at_threshold': float(np.mean(p > t)),
                    'human_flagged': int((n > t).sum()), 'human_fpr_ci95': clopper_pearson(int((n > t).sum()), len(n)),
                    'note': 'Only %d human papers: the 1%% threshold rests on ~%d papers.' % (len(n), int(np.floor(0.01 * len(n))) + 1)}
    # Fraction of sentences flagged at the sentence threshold, as a document score
    frac_h = np.array([float((cal[p] > t_sent_all).mean()) for p in cal if len(cal[p])])
    doc['sentence_flag_fraction'] = {'human_p50': float(np.median(frac_h)), 'human_p99': float(np.quantile(frac_h, .99)),
                                     'human_max': float(frac_h.max())}
    out = {'model': 'qwen35-4b fast10 (stage2-epoch0)', 'config': 'stride 510, LoRA merged, torch.compile, BF16',
           'human_set': 'paper-text-clean-v2-batch001, year<=2022, training-overlap removed',
           'ai_sets': {'sentence': 'eval suite paper_workflow_reconstruction edits (context view)',
                       'document': 'eval suite full_manuscripts (120, AI-involved)'},
           'sentence': sentence, 'document': doc}
    (HERE / 'results.json').write_text(json.dumps(out, indent=1))
    s = sentence
    print(json.dumps({'sentence_threshold': s['threshold_all_human'], 'heldout_fpr': s['heldout_fpr'], 'heldout_papers': s['heldout_papers'],
                      'tpr_at_1pct': s['tpr_at_threshold'], 'auc_vs_human_papers': {k: round(v['auc'], 4) for k, v in s['roc_vs_human_papers'].items()},
                      'doc': {k: {kk: v[kk] for kk in ('auc', 'threshold_1pct', 'tpr_at_threshold', 'human_flagged', 'human_fpr_ci95')} for k, v in doc.items() if 'auc' in v},
                      'sentence_flag_fraction': doc['sentence_flag_fraction']}, indent=1))


if __name__ == '__main__':
    main()
