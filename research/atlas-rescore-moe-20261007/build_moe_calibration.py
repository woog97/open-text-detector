"""Calibration page data for the final MoE, in the schema of app/lib/model-calibration/qwen35-4b-fast10.json (runs on the Space).

Aggregates only. Same rules as the 4B export (research/human-calibration-20261006/{analyze_calibration,export_calibration}.py
and research/evaluation/iclr2027-fast10-4b-20261005/analyze.py): sentence score = mean token AI probability; thresholds and
held-out checks from calibration-moe.json (cal_analyze.py); sentence ROC = eval-suite edited sentences (context view,
sets_for) against every human calibration sentence; document ROC = the suite's AI manuscripts against the human papers;
year rows = per-paper flagged share at the 1% threshold (human calibration groups, year diagnostic 2023-2026, ICLR 2027 Atlas).
Usage: build_moe_calibration.py OUT.json
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

D = Path('/data/workspace/atlas-rescore-moe-20261007'); sys.path.insert(0, str(D / 'code'))
import cal_analyze as ca  # noqa: E402
SUITE = Path('/data/workspace/evaluation-suite-v1-20261003'); HC = Path('/data/workspace/human-calibration-20261006')
CAL_TEXT = Path('/tmp/atlas-rescore/human-cal-clean'); OLD_TEXT = Path('/data/workspace/classifications/calibration-inputs/human-pre2023-batch001-v1')
YD_TEXT = Path('/data/workspace/classifications/calibration-inputs/year-diagnostic-v1')
FPR_GRID = np.unique(np.r_[np.geomspace(1e-4, 1, 81), np.linspace(0, 1, 101)])
EDIT = 'paper_workflow_reconstruction'; HUMAN_PAPER = {'human_paper_workflow_matched', 'human_paper_workflow_remaining'}
VENUE = {'iclr': 'ICLR', 'tmlr': 'TMLR', 'emnlp': 'EMNLP', 'corl': 'CoRL', 'neurips': 'NeurIPS'}
r5 = lambda v: round(float(v), 5)
rng = np.random.default_rng(20261007)


def sets_for(r):
    d, cond, view = r['dataset'], r.get('condition'), r.get('view'); pos = []
    if d == EDIT and r['label'] == 'mixed' and view == 'context':
        size = {'sentence': 'one', 'two_sentence': 'two', 'paragraph_concise': 'para', 'paragraph_v3': 'para'}[cond]
        pos = ['papers_all', f'papers_{size}'] + (['papers_small'] if size in ('one', 'two') else [])
    return pos


def roc(pos, neg, thr):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-s, kind='mergesort'); s, y = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0, np.cumsum(y)[last] / len(pos)]; fpr = np.r_[0, np.cumsum(1 - y)[last] / len(neg)]
    idx = np.searchsorted(fpr, FPR_GRID, side='right') - 1
    return {'fpr': [round(float(v), 6) for v in FPR_GRID], 'tpr': [round(float(v), 5) for v in tpr[idx]], 'auc': float(np.trapezoid(tpr, fpr)),
            'n_ai': int(len(pos)), 'n_human': int(len(neg)), 'tpr_at_threshold': float(np.mean(pos > thr))}


def curve(name, c): return {'name': name, 'auc': r5(c['auc']), 'n_ai': c['n_ai'], 'n_human': c['n_human'], 'tpr_at_1pct': r5(c['tpr_at_threshold']), 'fpr': c['fpr'], 'tpr': c['tpr']}


def clopper_pearson(k, n, a=0.05):
    """Exact binomial interval by bisection on the binomial tail (no scipy on the Space)."""
    from math import lgamma, exp, log
    def tail_ge(p, k):  # P(X >= k), X ~ Bin(n, p)
        if p <= 0: return 0.0 if k > 0 else 1.0
        if p >= 1: return 1.0
        return sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p)) for i in range(k, n + 1))
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) < target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    low = solve(lambda p: tail_ge(p, k), a / 2) if k > 0 else 0.0
    high = solve(lambda p: tail_ge(p, k + 1), 1 - a / 2) if k < n else 1.0
    return low, high


def doc_block(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; t = ca.threshold_at(v, 0.01); k = int((v > t).sum())
    reps = [ca.threshold_at(v[rng.integers(0, len(v), len(v))], 0.01) for _ in range(1000)]
    perm = rng.permutation(len(v)); A, B = v[perm[: len(v) // 2]], v[perm[len(v) // 2:]]; kb = int((B > ca.threshold_at(A, 0.01)).sum())
    lo, hi = clopper_pearson(kb, len(B))
    return {'threshold': r5(t), 'threshold_ci95': [r5(np.quantile(reps, .025)), r5(np.quantile(reps, .975))], 'human_fpr_ci95': [round(float(lo), 5), round(float(hi), 5)],
            'heldout_fpr': round(kb / len(B), 5), 'note': f'The 1% threshold rests on the top {k + 1} of {len(v)} human papers.'}


def shares_from(out_dir, text_dirs, t1):
    tok, docs = ca.load_tokens(out_dir); texts = {}; meta = {}
    for d in text_dirs:
        for f in sorted((d / 'data').glob('*.parquet')):
            cols = [c for c in ('id', 'text', 'conference', 'year', 'source') if c in pq.read_schema(f).names]
            for r in pq.read_table(f, columns=cols).to_pylist(): texts[r['id']] = r.pop('text'); meta[r['id']] = r
    out = {}
    for pid, z in tok.items():
        if pid not in texts: continue
        s = ca.sentence_scores(texts[pid], z['offsets'], z['probs'])
        out[pid] = {'share': float((s > t1).mean()) if len(s) else 0.0, 'scores': s, 'doc': docs.get(pid, {}).get('document_prob'), **meta[pid]}
    return out


def row(label, kind, year, ps, h99):
    sh = np.array([p['share'] for p in ps]); dc = [p['doc'] for p in ps if p['doc'] is not None]
    return {'label': label, 'kind': kind, 'year': year, 'papers': len(ps), 'flagged_share_median': r5(np.median(sh)), 'flagged_share_p25': r5(np.quantile(sh, .25)),
            'flagged_share_p75': r5(np.quantile(sh, .75)), 'over_human_p99': None if kind == 'human' else r5((sh > h99).mean()),
            'document_score_median': r5(np.median(dc)) if dc else None}


def main(out_path):
    cal = json.loads((D / 'calibration-moe.json').read_text()); v = cal['variants']['kept+old (as Atlas calibration)']
    t1 = next(t for t in v['thresholds'] if t['target_fpr'] == 0.01)['threshold']
    overlap = json.loads((HC / 'overlap.json').read_text())['papers']; meta = {p['id']: p for p in json.loads((HC / 'selection.json').read_text())['papers']}
    keep = {p for p, x in overlap.items() if x['shared_unique'] <= 0 and not x['key_hit']}
    human = shares_from(D / 'out-cal', [CAL_TEXT, OLD_TEXT], t1)
    old_ids = {r['id'] for f in sorted((OLD_TEXT / 'data').glob('*.parquet')) for r in pq.read_table(f, columns=['id']).to_pylist()}
    human = {p: x for p, x in human.items() if p in keep or p in old_ids}
    neg = np.concatenate([x['scores'] for x in human.values()]); h99 = float(np.quantile([x['share'] for x in human.values()], .99))
    # Suite positives: sentence_scores_space.suite() rows (per-sentence [score, gold]).
    suite = json.load(gzip.open(D / 'suite-sentences-moe.json.gz', 'rt')); pos = defaultdict(list)
    for r in suite:
        if 'sentences' not in r: continue
        for k in sets_for(r):
            pos[k] += [s for s, g in r['sentences'] if g == 1]
    sent_roc = {k: roc(pos[k], neg, t1) for k in ('papers_one', 'papers_two', 'papers_para')}
    # Document ROC: suite manuscripts (all AI-involved) vs human calibration papers.
    _, sdocs = ca.load_tokens(D / 'out-suite'); man = [r['id'] for r in suite if r.get('profile') == 'manuscripts']
    _, cdocs = ca.load_tokens(D / 'out-cal')
    hd = {c: np.array([cdocs[p][c] for p in human if cdocs.get(p, {}).get(c) is not None], float) for c in ('document_prob', 'token_prob_mean')}
    doc_roc = {}
    for c in ('document_prob', 'token_prob_mean'):
        p = np.array([sdocs[i][c] for i in man if i in sdocs and sdocs[i].get(c) is not None], float); t = ca.threshold_at(hd[c], 0.01)
        doc_roc[c] = roc(p, hd[c], t)
    dblock = doc_block(hd['document_prob'])
    # Year rows.
    rows = []
    for g in v['groups']:
        ps = [x for p, x in human.items() if (('archive', 2022) if p in old_ids else (meta[p]['venue'], meta[p]['year'])) == (g['venue'], g['year'])]
        label = f"{VENUE.get(g['venue'], g['venue'])} {g['year']}" if g['venue'] != 'archive' else 'Oct 5 set (archive, ≤2022)'
        rows.append(row(label, 'human', g['year'], ps, h99))
    yd = shares_from(D / 'out-yd', [YD_TEXT], t1); grp = defaultdict(list)
    for x in yd.values(): grp[(x['source'], x['conference'], x['year'])].append(x)
    for (src, venue, year), ps in sorted(grp.items()):
        s = {'archive': 'archive', 'openreview-2026-10': 'OpenReview download'}.get(src, src)
        rows.append(row(f'{VENUE.get(venue, venue)} {year}, {s}', 'iclr' if venue == 'iclr' else 'other', int(year), ps, h99))
    idx = pq.read_table(D / 'staging' / 'atlas-details-index-moe.parquet', columns=['pdf_sha256', 'score_summary_json']).to_pylist()
    atl_docs = {}
    for o in ('out-space', 'out-h200'):
        for f in sorted((D / o / 'data').glob('summary-*.parquet')):
            for r in pq.read_table(f, columns=['pdf_sha256', 'document_prob']).to_pylist(): atl_docs[r['pdf_sha256']] = r['document_prob']
    atl = [{'share': (lambda s: s['flagged'] / max(s['total'], 1))(json.loads(r['score_summary_json'])), 'doc': atl_docs.get(r['pdf_sha256'])} for r in idx if r['score_summary_json']]
    rows.append(row('ICLR 2027 submissions', 'iclr', 2027, atl, h99))
    rows = sorted([r for r in rows if r['kind'] == 'human'], key=lambda g: (g['year'], g['label'])) + \
           sorted([r for r in rows if r['kind'] != 'human'], key=lambda g: (g['year'], g['label']))
    ov = json.loads((HC / 'overlap.json').read_text()); n_kept = sum(1 for p in human if p in keep)
    data = {
        'model': {'id': 'qwen36-35b-a3b', 'name': 'Qwen3.6 35B-A3B MoE · Oct 7',
                  'detail': ('LoRA classifier on Qwen3.6-35B-A3B (mixture of experts), the final run of Oct 7, 2026: one pass over a 72,000-row training mix '
                             'with small edits from Claude, Luna, GPT-6.1 Sol and DeepSeek, paragraph edits, AI sections and human paper text (last checkpoint, '
                             'stage2-epoch2); 510-token windows without overlap, LoRA merged, BF16.')},
        'calibration_set': {'papers': v['n_papers'], 'sentences': v['n_sentences'],
                            'description': (f"{v['n_papers']:,} human papers (human by project rule: 2022 or earlier). {n_kept:,} are ICLR 2019–2022, NeurIPS 2021–2022 "
                                            f"and CoRL 2021–2022 papers sampled by venue and year from the site's PDF collection, after removing those that share text "
                                            f"with a scored model’s training data; {v['n_papers'] - n_kept} are the 2020–2022 archive papers used before October 7. "
                                            f"The same papers calibrate the Qwen3.5-4B model."),
                            'method': ('Thresholds are set on all human papers. Each is checked by setting it on a random half of the papers and measuring the flag rate on '
                                       'the other half. Flags cluster within papers, so every interval resamples whole papers rather than treating sentences as independent.')},
        'sentence_score': 'Mean AI probability of the tokens in the sentence (raw model output, not rescaled).',
        'thresholds': [{'target_fpr': t['target_fpr'], 'threshold': r5(t['threshold']), 'threshold_ci95': [r5(x) for x in t['threshold_ci95']],
                        'heldout_fpr': round(t['heldout_fpr'], 6), 'heldout_ci95': [round(x, 6) for x in t['heldout_ci95_paper_bootstrap']],
                        'heldout_sentences': t['heldout_sentences'], 'heldout_papers': t['heldout_papers'],
                        'stable': t['heldout_ci95_paper_bootstrap'][0] <= t['target_fpr'] <= t['heldout_ci95_paper_bootstrap'][1] and t['heldout_ci95_paper_bootstrap'][1] <= 2 * t['target_fpr']}
                       for t in v['thresholds']],
        'sentence_curves': [curve('One-sentence edits', sent_roc['papers_one']), curve('Two-sentence edits', sent_roc['papers_two']), curve('Paragraph edits', sent_roc['papers_para'])],
        'document_curves': [curve('Document head', doc_roc['document_prob']), curve('Mean token score', doc_roc['token_prob_mean'])],
        'document_threshold': dblock,
        'years': rows,
        'candidates': [],
        'caveats': [f"Thresholds rest on {v['n_papers']:,} human papers from 2019–2022, mostly machine learning venues. Topics and style differ from 2025–2027 writing, so false-positive rates on recent human papers may be higher than shown.",
                    'Flags measure resemblance to AI-generated text. They cannot separate AI use from human writing that has adopted LLM-like phrasing.',
                    'The 1% threshold is very low (raw sentence probability about 0.006): this model gives human sentences scores near zero, so moderately scored sentences are flagged. Very short or badly extracted texts can therefore show high flagged shares.',
                    'One training run (one seed). Classifications are model outputs, not conclusions about authorship.']}
    Path(out_path).write_text(json.dumps(data, separators=(',', ':')))
    print(json.dumps({'t1': t1, 'sentence_auc': {k: round(x['auc'], 4) for k, x in sent_roc.items()}, 'tpr_at_1pct': {k: round(x['tpr_at_threshold'], 4) for k, x in sent_roc.items()},
                      'n_pos': {k: x['n_ai'] for k, x in sent_roc.items()}, 'n_neg': int(len(neg)), 'doc': {c: (round(x['auc'], 4), x['n_ai'], x['n_human']) for c, x in doc_roc.items()},
                      'doc_threshold': dblock, 'years': [(r['label'], r['papers'], r['flagged_share_median']) for r in rows]}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
