"""Recompute 1%-FPR thresholds from the expanded human calibration set, with paper-clustered bootstrap intervals.

Usage: analyze_calibration.py SCORES_DIR OLD_CACHE OUT.json
  SCORES_DIR/<model>/{sentences.npz,documents.parquet}  new human papers scored by score_h200.py
  SCORES_DIR/overlap.json, SCORES_DIR/selection.json    training-overlap counts and paper metadata
  OLD_CACHE   the Oct 5 calibration cache (cal-sentences.npz, cal-summary.parquet: 215 papers scored by the Atlas
              scorer; suite-sentences.json.gz, suite-summary.parquet: AI positives). Pooled into the Atlas model only,
              since only it was scored on them.
Flags cluster within papers, so every interval resamples papers, not sentences. Writes aggregates only.
"""
import gzip, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
import importlib.util
sys.path.insert(0, str(HERE.parent / 'evaluation/backbone-results-20261003'))
from sentence_roc import sets_for  # noqa: E402
# The Oct 5 analysis (same ROC grid, threshold rule, half split, Clopper-Pearson); loaded by path, as several modules are named analyze.
_spec = importlib.util.spec_from_file_location('oct5_analyze', HERE.parent / 'evaluation/iclr2027-fast10-4b-20261005/analyze.py')
oct5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(oct5)
roc, threshold_at, half, clopper_pearson = oct5.roc, oct5.threshold_at, oct5.half, oct5.clopper_pearson

TARGETS = [0.001, 0.01, 0.05]
B = 1000
MAX_SHARED = 0  # excluded if any shared 10+-word sentence occurs in no other calibration paper (cited titles are shared widely)
rng = np.random.default_rng(20261006)


def load_new(model_dir, keep):
    z = np.load(model_dir / 'sentences.npz'); cal = {}
    for k in z.files:
        pid, kind = k.rsplit('/', 1)
        if kind == 'scores' and pid in keep: cal[pid] = z[k].astype(float)
    docs = pq.read_table(model_dir / 'documents.parquet').to_pandas().set_index('id')
    return cal, docs.loc[[p for p in docs.index if p in keep]]


def weighted_threshold(scores, owner, weights, fpr):
    """Threshold with flag rate <= fpr when paper i's sentences count weights[i] times (bootstrap replicate)."""
    order = np.argsort(-scores, kind='mergesort'); w = weights[owner[order]]
    cum = np.cumsum(w); total = cum[-1]
    i = np.searchsorted(cum, np.floor(fpr * total), side='right')
    return float(scores[order][min(i, len(order) - 1)])


def paper_boot_rate(flags, counts, n=B):
    """Paper-clustered bootstrap of a pooled rate sum(flags)/sum(counts)."""
    idx = rng.integers(0, len(flags), size=(n, len(flags)))
    rates = flags[idx].sum(1) / np.maximum(counts[idx].sum(1), 1)
    return [float(np.quantile(rates, .025)), float(np.quantile(rates, .975))]


def sentence_block(cal):
    pids = sorted(cal); scores = np.concatenate([cal[p] for p in pids])
    owner = np.concatenate([np.full(len(cal[p]), i) for i, p in enumerate(pids)])
    counts = np.array([len(cal[p]) for p in pids], float)
    split = np.array([half(p) for p in pids])
    out = {'n_papers': len(pids), 'n_sentences': int(len(scores)), 'thresholds': []}
    for f in TARGETS:
        t = threshold_at(scores, f)
        # Threshold uncertainty: re-pick it on paper-bootstrap replicates (multinomial paper weights).
        reps = [weighted_threshold(scores, owner, np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids)).astype(float), f)
                for _ in range(B // 4)]
        # Held-out check: threshold from half A, flag rate on half B, interval resampling B's papers.
        a = np.concatenate([cal[p] for p, s in zip(pids, split) if s == 0]); ta = threshold_at(a, f)
        fb = np.array([(cal[p] > ta).sum() for p, s in zip(pids, split) if s == 1], float)
        cb = np.array([len(cal[p]) for p, s in zip(pids, split) if s == 1], float)
        flags = np.array([(cal[p] > t).sum() for p in pids], float)
        out['thresholds'].append({
            'target_fpr': f, 'threshold': t, 'threshold_ci95': [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))],
            'heldout_threshold_half_a': ta, 'heldout_fpr': float(fb.sum() / cb.sum()), 'heldout_ci95_paper_bootstrap': paper_boot_rate(fb, cb),
            'heldout_papers': int(len(fb)), 'heldout_sentences': int(cb.sum()),
            'papers_with_any_flag': float(np.mean(flags > 0)), 'per_paper_flag_share_p50': float(np.median(flags / np.maximum(counts, 1))),
            'per_paper_flag_share_p99': float(np.quantile(flags / np.maximum(counts, 1), .99)),
            # What sentence-independent (Clopper-Pearson style) intervals would claim, for comparison.
            'naive_binomial_halfwidth': float(1.96 * np.sqrt(f * (1 - f) / cb.sum()))})
    t1 = next(x['threshold'] for x in out['thresholds'] if x['target_fpr'] == 0.01)
    share = {p: float((cal[p] > t1).mean()) if len(cal[p]) else 0.0 for p in pids}
    return out, share, t1


def document_block(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    t = threshold_at(v, 0.01); k = int((v > t).sum())
    reps = [threshold_at(v[rng.integers(0, len(v), len(v))], 0.01) for _ in range(B)]
    # Held-out: threshold on a random half of papers, rate on the other half (papers are the unit here).
    perm = rng.permutation(len(v)); A, Bh = v[perm[: len(v) // 2]], v[perm[len(v) // 2:]]
    tb = threshold_at(A, 0.01); kb = int((Bh > tb).sum())
    return {'n_papers': int(len(v)), 'threshold_1pct': t, 'threshold_ci95': [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))],
            'papers_above': k, 'heldout_fpr': kb / len(Bh), 'heldout_ci95': clopper_pearson(kb, len(Bh)), 'heldout_papers': int(len(Bh)),
            'note': 'The 1%% threshold rests on the top %d of %d human papers.' % (k + 1, len(v))}


def main(scores_dir, old_cache, out_path):
    scores_dir, old_cache = Path(scores_dir), Path(old_cache)
    overlap = json.loads((scores_dir / 'overlap.json').read_text())['papers']
    meta = {p['id']: p for p in json.loads((scores_dir / 'selection.json').read_text())['papers']}
    keep = {p for p, v in overlap.items() if v['shared_unique'] <= MAX_SHARED and not v['key_hit']}
    result = {'overlap': {'scored': len(overlap), 'excluded': len(overlap) - len(keep), 'kept': len(keep),
                          'rule': 'excluded if the OpenReview id, or any normalized 10+-word sentence that occurs in no other calibration paper, appears in a scored model\'s training, selection or calibration windows, the splice pool or the sweep-eval rows (sentences shared by several calibration papers are cited reference titles)'},
              'models': {}}
    for model_dir in sorted(p for p in scores_dir.iterdir() if (p / 'sentences.npz').exists()):
        cal, docs = load_new(model_dir, keep)
        pooled = model_dir.name == 'atlas-fast10'
        if pooled:  # add the Oct 5 set (same scorer, same model, disjoint papers by construction of the selection)
            z = np.load(old_cache / 'cal-sentences.npz')
            for k in z.files:
                pid, kind = k.rsplit('/', 1)
                if kind == 'scores': cal['old:' + pid] = z[k].astype(float)
            old_docs = pq.read_table(old_cache / 'cal-summary.parquet').to_pandas()
            doc_vals = {c: np.r_[docs[c].to_numpy(float), old_docs[c].to_numpy(float)] for c in ('document_prob', 'token_prob_mean')}
        else:
            doc_vals = {c: docs[c].to_numpy(float) for c in ('document_prob', 'token_prob_mean')}
        sent, share, t1 = sentence_block(cal)
        m = {'sentence': sent, 'document': {c: document_block(v) for c, v in doc_vals.items()},
             'includes_oct5_set': pooled}
        # Per venue/year (new papers only): flagged share at the pooled 1% threshold.
        groups = defaultdict(list)
        for p, s in share.items():
            if p.startswith('old:'): groups[('archive', 2022)].append(s); continue
            groups[(meta[p]['venue'], meta[p]['year'])].append(s)
        docp = docs['document_prob'].to_dict()
        m['groups'] = [{'venue': v, 'year': y, 'papers': len(s), 'flagged_share_mean': float(np.mean(s)), 'flagged_share_p50': float(np.median(s)),
                        'flagged_share_p25': float(np.quantile(s, .25)), 'flagged_share_p75': float(np.quantile(s, .75)),
                        'document_prob_p50': float(np.median([docp[p] for p in docp if meta[p]['venue'] == v and meta[p]['year'] == y])) if v != 'archive' else None}
                       for (v, y), s in sorted(groups.items())]
        m['human_flag_share_hist'] = np.histogram(list(share.values()), bins=20, range=(0, 0.2))[0].tolist()
        if pooled:  # ROC vs the pooled human negatives, with the eval-suite AI positives
            suite = json.load(gzip.open(old_cache / 'suite-sentences.json.gz', 'rt'))
            pos = defaultdict(list)
            for r in suite:
                if 'sentences' not in r: continue
                p_sets, _ = sets_for(r)
                for score, gold in r['sentences']:
                    if gold == 1:
                        for k in p_sets: pos[k].append(score)
            neg = np.concatenate(list(cal.values()))
            m['sentence_roc'] = {k: {**roc(pos[k], neg), 'tpr_at_threshold': float(np.mean(np.asarray(pos[k]) > t1))}
                                 for k in ('papers_one', 'papers_two', 'papers_para')}
            sd = pq.read_table(old_cache / 'suite-summary.parquet').to_pandas().set_index('id')
            ai = [r['id'] for r in suite if r['profile'] == 'manuscripts']
            m['document_roc'] = {}
            for c in ('document_prob', 'token_prob_mean'):
                p = sd.loc[ai, c].to_numpy(float); t = m['document'][c]['threshold_1pct']
                m['document_roc'][c] = {**roc(p, doc_vals[c]), 'tpr_at_threshold': float(np.mean(p > t))}
        result['models'][model_dir.name] = m
        print(model_dir.name, json.dumps({'papers': sent['n_papers'], 'sentences': sent['n_sentences'],
                                          'thresholds': [{k: x[k] for k in ('target_fpr', 'threshold', 'threshold_ci95', 'heldout_fpr', 'heldout_ci95_paper_bootstrap', 'naive_binomial_halfwidth')} for x in sent['thresholds']],
                                          'doc': {c: {k: v[k] for k in ('n_papers', 'threshold_1pct', 'threshold_ci95', 'heldout_fpr', 'heldout_ci95')} for c, v in m['document'].items()}}, indent=1), flush=True)
    Path(out_path).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
