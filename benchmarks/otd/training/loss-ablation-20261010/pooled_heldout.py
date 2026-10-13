"""Pooled small-edit recall at 1% FPR over the three held-out sets, with a paired paper bootstrap between runs.

Reads what eval_final.py saves for each set: <set>-<ck>-sentences.npz (per document, one (score, label) row per sentence;
label 1 = AI, 0 = human, -1 = unlabelled) and <set>-<ck>-rows.json (id, slot, condition, writer per document). Definitions
match cross_model_eval.py, so per-set numbers reproduce its recall_at_1pct_controls:
  - controls: human-labelled sentences in the human-original documents (slot human_controls);
  - threshold per set: the 1%-FPR cutoff on that set's controls, sort(desc)[floor(0.01 n)]; flagged = score > cutoff;
  - small-edit positives: AI sentences of mixed documents in the cohorts insert_one, rewrite_one, rewrite_two, split_one;
  - spill-over: share of human sentences inside those edited documents that are flagged (false positives beside an edit).
Pooled recall = flagged small-edit sentences / all small-edit sentences, each set at its own cutoff.
Bootstrap: resample papers (id prefix before the first '/') with replacement within each set, the same draw for every run,
and recompute cutoffs and recalls; differences against --ref get a 95% percentile interval.

  python pooled_heldout.py --run NAME=EVAL_DIR[:CHECKPOINT] ... [--ref NAME] [--boot 1000] [--out results.json]
CHECKPOINT defaults to the only <set>-*-sentences.npz checkpoint in the folder (e.g. stage2-epoch0).
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np

SETS = ['heldout-writers', 't21-heldout-v2', 'wide1-heldout']
SMALL = ['insert_one', 'rewrite_one', 'rewrite_two', 'split_one']


def cutoff(neg):
    return np.sort(neg)[::-1][int(math.floor(.01 * len(neg)))]


def load(d, ck, s):
    if ck is None:
        cks = sorted({c for c in (os.path.basename(p)[len(s) + 1:-len('-sentences.npz')] for p in glob.glob(os.path.join(d, f'{s}-*-sentences.npz'))) if re.fullmatch(r'stage\d-epoch\d+|step\d+', c)})
        assert len(cks) == 1, f'{d} {s}: pick a checkpoint from {cks}'
        ck = cks[0]
    rows = json.load(open(os.path.join(d, f'{s}-{ck}-rows.json'))); z = np.load(os.path.join(d, f'{s}-{ck}-sentences.npz'))
    return ck, rows, [z[str(i)] for i in range(len(rows))]


def flatten(rows, sents):
    """Per-sentence arrays for one set: score, kind (0 control, 1 small-edit AI, 2 human beside a small edit), paper, cohort, writer."""
    out = defaultdict(list)
    for r, x in zip(rows, sents):
        paper = r['id'].split('/')[0]
        for sc, lab in x:
            if r['slot'] == 'human_controls' and lab == 0: k = 0
            elif r['slot'] == 'mixed' and r['condition'] in SMALL and lab == 1: k = 1
            elif r['slot'] == 'mixed' and r['condition'] in SMALL and lab == 0: k = 2
            else: continue
            out['score'].append(sc); out['kind'].append(k); out['paper'].append(paper); out['cohort'].append(r['condition']); out['writer'].append(r['writer'])
    return {k: np.asarray(v) for k, v in out.items()}


def metrics(score, kind, cohort, writer):
    t = cutoff(score[kind == 0]); f = score > t; pos = kind == 1; sp = kind == 2
    m = {'cutoff': float(t), 'n_controls': int((kind == 0).sum()), 'n_small': int(pos.sum()), 'flagged_small': int(f[pos].sum()),
         'recall': float(f[pos].mean()), 'n_spill': int(sp.sum()), 'spill_fpr': float(f[sp].mean()) if sp.any() else None}
    m['cohort'] = {c: {'n': int((pos & (cohort == c)).sum()), 'flagged': int(f[pos & (cohort == c)].sum())} for c in SMALL}
    m['writer'] = {w: {'n': int((pos & (writer == w)).sum()), 'flagged': int(f[pos & (writer == w)].sum())} for w in sorted(set(writer[pos]))}
    return m


def pooled(per_set):
    n = sum(m['n_small'] for m in per_set.values()); k = sum(m['flagged_small'] for m in per_set.values())
    ns = sum(m['n_spill'] for m in per_set.values()); ks = sum(round(m['spill_fpr'] * m['n_spill']) for m in per_set.values() if m['n_spill'])
    coh = {c: sum(m['cohort'][c]['n'] for m in per_set.values()) for c in SMALL}
    cf = {c: sum(m['cohort'][c]['flagged'] for m in per_set.values()) for c in SMALL}
    wr = defaultdict(lambda: [0, 0])
    for m in per_set.values():
        for w, v in m['writer'].items(): wr[w][0] += v['n']; wr[w][1] += v['flagged']
    return {'n_small': n, 'recall': k / n, 'n_spill': ns, 'spill_fpr': ks / ns if ns else None,
            'cohort': {c: {'n': coh[c], 'recall': cf[c] / coh[c] if coh[c] else None} for c in SMALL},
            'writer': {w: {'n': v[0], 'recall': v[1] / v[0]} for w, v in sorted(wr.items())}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', action='append', required=True); ap.add_argument('--ref')
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out')
    ap.add_argument('--suffix', default='', help="set-name suffix, e.g. '-lc' for the long-context sets")
    ap.add_argument('--check', action='store_true', help='compare per-set cohort recalls with the saved <set>-<ck>.json')
    a = ap.parse_args()
    global SETS
    SETS = [s + a.suffix for s in SETS]
    runs = {}
    for spec in a.run:
        name, rest = spec.split('=', 1); d, _, ck = rest.partition(':'); runs[name] = (d, ck or None)
    data = {s: {} for s in SETS}; meta = {}
    for name, (d, ck) in runs.items():
        for s in SETS:
            c, rows, sents = load(d, ck, s); data[s][name] = (rows, flatten(rows, sents)); meta[(name, s)] = (d, c)
    for s in SETS:  # every run must have scored the same documents in the same order
        ids = {name: [r['id'] for r in rows] for name, (rows, _) in data[s].items()}
        first = next(iter(ids.values())); assert all(v == first for v in ids.values()), f'{s}: runs scored different rows'
    res = {'definition': __doc__.split('\n\n')[0], 'sets': SETS, 'small_cohorts': SMALL, 'runs': {}}
    for name in runs:
        per_set = {s: metrics(*(data[s][name][1][k] for k in ('score', 'kind', 'cohort', 'writer'))) for s in SETS}
        res['runs'][name] = {'checkpoints': {s: meta[(name, s)][1] for s in SETS}, 'per_set': per_set, 'pooled': pooled(per_set)}
        if a.check:
            for s in SETS:
                d, c = meta[(name, s)]; saved = json.load(open(os.path.join(d, f'{s}-{c}.json')))['groups']
                for coh in SMALL:
                    g = saved.get(f'cohort:{coh}')
                    if g is None: continue
                    mine = per_set[s]['cohort'][coh]; mine = mine['flagged'] / mine['n']
                    assert abs(mine - g['recall_at_1pct_controls']) < 1e-9 and mine is not None, (name, s, coh, mine, g['recall_at_1pct_controls'])
            print(f'check ok: {name} per-set cohort recalls match recall_at_1pct_controls', file=sys.stderr)
    if a.boot and len(runs) > 1:
        ref = a.ref or next(iter(runs)); rng = np.random.default_rng(a.seed); names = list(runs)
        # per set: sentence indices grouped by paper (identical across runs, checked above)
        groups = {}
        for s in SETS:
            papers = data[s][names[0]][1]['paper']; order = np.argsort(papers, kind='stable'); u, start = np.unique(papers[order], return_index=True)
            groups[s] = np.split(order, start[1:])
        draws = {n: {'recall': [], 'spill_fpr': []} for n in names}
        for _ in range(a.boot):
            idx = {s: np.concatenate([groups[s][i] for i in rng.integers(0, len(groups[s]), len(groups[s]))]) for s in SETS}
            for n in names:
                k = tot = ks = ts = 0
                for s in SETS:
                    f = data[s][n][1]; sc, kd = f['score'][idx[s]], f['kind'][idx[s]]; fl = sc > cutoff(sc[kd == 0])
                    k += fl[kd == 1].sum(); tot += (kd == 1).sum(); ks += fl[kd == 2].sum(); ts += (kd == 2).sum()
                draws[n]['recall'].append(k / tot); draws[n]['spill_fpr'].append(ks / ts)
        res['bootstrap'] = {'reference': ref, 'replicates': a.boot, 'unit': 'paper, resampled within each set', 'runs': {}}
        for n in names:
            e = {}
            for mname in ('recall', 'spill_fpr'):
                v = np.asarray(draws[n][mname]); dv = v - np.asarray(draws[ref][mname])
                e[mname] = {'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                if n != ref:
                    e[mname]['diff_vs_ref'] = res['runs'][n]['pooled'][mname] - res['runs'][ref]['pooled'][mname]
                    e[mname]['diff_ci95'] = [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))]
                    e[mname]['share_of_draws_below_ref'] = float((dv < 0).mean())
            res['bootstrap']['runs'][n] = e
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    w = max(len(n) for n in runs)
    print(f"{'run':<{w}}  small-edit recall@1%  [95% CI]        spill FPR   " + '  '.join(f'{c:>11}' for c in SMALL) + ('   diff vs ' + (a.ref or names[0]) if 'bootstrap' in res else ''))
    for n in runs:
        p = res['runs'][n]['pooled']; b = res.get('bootstrap', {}).get('runs', {}).get(n, {})
        ci = b.get('recall', {}).get('ci95'); d = b.get('recall', {})
        line = f"{n:<{w}}  {p['recall']:.3f} (n={p['n_small']})  " + (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ' ' * 16)
        line += f"  {p['spill_fpr']:.4f}    " + '  '.join(f"{p['cohort'][c]['recall']:>11.3f}" for c in SMALL)
        if 'diff_vs_ref' in d: line += f"   {d['diff_vs_ref']:+.3f} [{d['diff_ci95'][0]:+.3f}, {d['diff_ci95'][1]:+.3f}]"
        print(line)


if __name__ == '__main__':
    main()
