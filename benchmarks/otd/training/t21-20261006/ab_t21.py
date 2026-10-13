"""Space side: T2.1 mix decision tables. Per arm (3 seeds each, stage2-epoch2) and evaluation set: recall at the paper
threshold (1% FPR on sweep-eval dev human sentences, fixed per seed) overall / per writer / per source, human-control FPR
at that threshold and AUROC; plus a paired bootstrap (papers resampled, 2,000 draws) of seed-mean differences between arms
on the rows both arms were scored on. Sweep eval: test/dev small-edit recall at each seed's own 1% FPR threshold.
SPG on v2 drops rows that overlap its splices (v2-spg-overlap-exclude.json). Usage: ab_t21.py OUT.json"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'; CK = 'stage2-epoch2'; rng = np.random.default_rng(20261006)
ARMS = {a: [f'q4b-{a}-s{s}' for s in (1, 2, 3)] for a in ('T21', 'T21S', 'T2', 'SPG')}
V2 = {r['id']: r for r in map(json.loads, gzip.open(R / 'inputs/t21-heldout-score-rows-v2.jsonl.gz', 'rt'))}
SPG_EX = set(json.load(open(R / 'inputs/v2-spg-overlap-exclude.json'))['ids'])
SW = [json.loads(l) for l in gzip.open(S / 'sweep-eval-rows.jsonl.gz', 'rt')]


def sweep_sents(tag):
    z = np.load(S / tag / 'eval' / f'{CK}-sentences.npz'); return [z[str(i)] for i in range(len(SW))]


def thr_in(sents):
    neg = np.concatenate([x[x[:, 1] == 0, 0] for r, x in zip(SW, sents) if r['split'] == 'dev' and r['slot'] in ('edit_context', 'human_untouched', 'human_paper')])
    return float(np.sort(neg)[::-1][int(np.floor(.01 * len(neg)))])


def auroc(pos, neg):
    if len(pos) < 5 or len(neg) < 5: return None
    s = np.r_[pos, neg]; o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(tag, name):
    f = S / tag / 'eval' / f'{name}-{CK}-sentences.npz'
    if not f.exists(): return None
    z = np.load(f); rows = json.loads((S / tag / 'eval' / f'{name}-{CK}-rows.json').read_text())
    return {r['id']: (r, z[str(i)]) for i, r in enumerate(rows)}


def tally(data, ids, thr, key):
    """Per-row hit counts for AI sentences and false-positive counts for human-control sentences at thr."""
    out = {}
    for i in ids:
        r, x = data[i]; pos = x[x[:, 1] == 1, 0]; neg = x[x[:, 1] == 0, 0] if r['slot'] == 'human_controls' else np.zeros(0)
        flag = int(len(x) > 0 and x[:, 0].max() > thr)  # document flagged: any sentence (labelled or mixed) above thr
        extra = {('docflag_ai' if r['slot'] == 'mixed' else 'docflag_controls'): (flag, 1)}
        if r['slot'] == 'human_controls':  # human-control sentence FPR per source / writer group
            extra.update({'fp|' + k: ((neg > thr).sum(), len(neg)) for k in key(r)})
        out[i] = (key(r), (pos > thr).sum(), len(pos), (neg > thr).sum(), len(neg), extra)
    return out


def nmean(v):
    v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None


def group_key(i):  # paper (or host) id; the Fable sets prefix ids with the set name
    p = i.split('/'); return p[1] if p[0] in ('polish', 'humanizer', 'cowrite') and len(p) > 1 else p[0]


def summarize(name, arms_ids, key):
    """arms_ids: {arm: (tags, ids)}. Returns per-arm metrics and a paired bootstrap on the common rows."""
    res = {}; per_arm_rows = {}
    for arm, (tags, ids) in arms_ids.items():
        seeds = []; tall = []
        for t in tags:
            d = load(t, name)
            if d is None: continue
            ids_t = [i for i in ids if i in d]
            if not ids_t: continue
            th = thr_in(sweep_sents(t)); tl = tally(d, ids_t, th, key); tall.append(tl)
            g = defaultdict(lambda: [0, 0]); fp = [0, 0]
            for i, (k, h, n, f, m, ex) in tl.items():
                for kk in ('all',) + tuple(k):
                    g[kk][0] += h; g[kk][1] += n
                for kk, (h2, n2) in ex.items():
                    g[kk][0] += h2; g[kk][1] += n2
                fp[0] += f; fp[1] += m
            pos = np.concatenate([d[i][1][d[i][1][:, 1] == 1, 0] for i in ids_t] + [np.zeros(0)]); neg = np.concatenate([d[i][1][d[i][1][:, 1] == 0, 0] for i in ids_t if d[i][0]['slot'] == 'human_controls'] + [np.zeros(0)])
            seeds.append({'tag': t, 'threshold': th, 'n_rows': len(ids_t), 'recall': {k: v[0] / v[1] for k, v in g.items() if v[1]}, 'n_ai': {k: v[1] for k, v in g.items()},
                          'control_fpr': fp[0] / fp[1] if fp[1] else None, 'auroc_vs_controls': auroc(pos, neg)})
        if seeds:
            res[arm] = {'seeds': seeds, 'mean_recall': {k: float(np.mean([s['recall'][k] for s in seeds if k in s['recall']])) for k in seeds[0]['recall']},
                        'mean_control_fpr': nmean([s['control_fpr'] for s in seeds]), 'mean_auroc': nmean([s['auroc_vs_controls'] for s in seeds])}
            per_arm_rows[arm] = tall
    boots = {}
    arms = list(per_arm_rows)
    for a in arms:
        for b in arms:
            if a >= b or len(per_arm_rows[a]) < 2 or len(per_arm_rows[b]) < 2: continue
            common = set.intersection(*[set(t) for t in per_arm_rows[a] + per_arm_rows[b]])
            papers = sorted({group_key(i) for i in common}); pidx = {p: j for j, p in enumerate(papers)}
            keys = sorted({kk for t in per_arm_rows[a] for i in common for kk in ('all',) + tuple(t[i][0]) + tuple(t[i][5])})
            def mat(tl_list):  # [seed][key] -> per-paper (hits, n), plus fp
                H = np.zeros((len(tl_list), len(keys), len(papers))); N = np.zeros_like(H); F = np.zeros((len(tl_list), len(papers))); M = np.zeros_like(F)
                for s, tl in enumerate(tl_list):
                    for i in common:
                        k, h, n, f, m, ex = tl[i]; p = pidx[group_key(i)]; F[s, p] += f; M[s, p] += m
                        for kk in ('all',) + tuple(k):
                            H[s, keys.index(kk), p] += h; N[s, keys.index(kk), p] += n
                        for kk, (h2, n2) in ex.items():
                            H[s, keys.index(kk), p] += h2; N[s, keys.index(kk), p] += n2
                return H, N, F, M
            A, B = mat(per_arm_rows[a]), mat(per_arm_rows[b])
            W = rng.multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=2000).T  # papers x draws
            def rec(X): H, N, F, M = X; return ((H @ W) / np.maximum(N @ W, 1)).mean(0), ((F @ W) / np.maximum(M @ W, 1)).mean(0)
            (ra, fa), (rb, fb) = rec(A), rec(B)
            pt = lambda X: ((X[0].sum(2) / np.maximum(X[1].sum(2), 1)).mean(0), (X[2].sum(1) / np.maximum(X[3].sum(1), 1)).mean())
            (pa, fpa), (pb, fpb) = pt(A), pt(B); dr = rb - ra; df = fb - fa
            boots[f'{b}-{a}'] = {'n_common_rows': len(common), 'n_papers': len(papers),
                                 'recall_diff': {k: {'diff': float(pb[j] - pa[j]), 'ci95': [float(np.percentile(dr[j], 2.5)), float(np.percentile(dr[j], 97.5))],
                                                     f'{a}': float(pa[j]), f'{b}': float(pb[j]), 'n_ai': int(A[1][0, j].sum())} for j, k in enumerate(keys)},
                                 'control_fpr_diff': {'diff': float(fpb - fpa), 'ci95': [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))]}}
    return {'arms': res, 'paired': boots}


def sweep_table():
    out = {}; per = {}
    for arm, tags in ARMS.items():
        for t in tags:
            if not (S / t / 'eval' / f'{CK}-sentences.npz').exists(): continue
            x = sweep_sents(t)
            for split in ('dev', 'test'):
                neg = np.concatenate([s[s[:, 1] == 0, 0] for r, s in zip(SW, x) if r['split'] == split and r['slot'] in ('edit_context', 'human_untouched', 'human_paper')])
                th = float(np.sort(neg)[::-1][int(np.floor(.01 * len(neg)))])
                tl = {}
                for i, (r, s) in enumerate(zip(SW, x)):
                    if r['split'] == split and r['slot'] == 'edit_context' and r['condition'] in ('sentence', 'two_sentence'):
                        pos = s[s[:, 1] == 1, 0]; tl[i] = (str(r.get('group_id') or r['id'].split('/')[0]), (pos > th).sum(), len(pos))
                per.setdefault((arm, split), []).append(tl)
    for split in ('dev', 'test'):
        for a in ARMS:
            for b in ARMS:
                if a >= b or len(per.get((a, split), [])) < 2 or len(per.get((b, split), [])) < 2: continue
                rows = sorted(per[(a, split)][0]); papers = sorted({per[(a, split)][0][i][0] for i in rows}); pidx = {p: j for j, p in enumerate(papers)}
                def mat(L):
                    H = np.zeros((len(L), len(papers))); N = np.zeros_like(H)
                    for s, tl in enumerate(L):
                        for i in rows: H[s, pidx[tl[i][0]]] += tl[i][1]; N[s, pidx[tl[i][0]]] += tl[i][2]
                    return H, N
                (HA, NA), (HB, NB) = mat(per[(a, split)]), mat(per[(b, split)])
                W = rng.multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=2000).T
                d = ((HB @ W) / (NB @ W)).mean(0) - ((HA @ W) / (NA @ W)).mean(0)
                out[f'{split}:{b}-{a}'] = {a: float((HA.sum(1) / NA.sum(1)).mean()), b: float((HB.sum(1) / NB.sum(1)).mean()), 'diff': float((HB.sum(1) / NB.sum(1)).mean() - (HA.sum(1) / NA.sum(1)).mean()),
                                           'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], 'n_papers': len(papers), 'n_sentences': int(NA[0].sum())}
    return out


wkey = lambda r: ('w:' + r['writer'],) if r['slot'] == 'mixed' else ()
v2key = lambda r: (('w:' + r['writer'],) if r['slot'] == 'mixed' else ()) + ('src:' + V2.get(r['id'], {}).get('source', '?') + (':ai' if r['slot'] == 'mixed' else ':human'),)
clean = [i for i, r in V2.items() if str(r.get('t21_may_have_seen_paper')) != 'True']
spg_clean = [i for i in clean if i not in SPG_EX]
OUT = {'sweep_small_edits': sweep_table()}
for name in ('heldout-writers', 'cross-model-s500', 'fable-polish', 'fable-humanizer', 'fable-cowrite'):
    ids = None
    for t in ARMS['T21']:
        d = load(t, name)
        if d: ids = sorted(d); break
    if ids: OUT[name] = summarize(name, {a: (tags, ids) for a, tags in ARMS.items()}, wkey)
# v2: T2.1 arms on the clean rows; SPG on clean minus splice overlap (T2.1 arms also reported on that subset); T2 on clean rows (not clean for T2).
OUT['v2-clean'] = summarize('t21-heldout-v2-clean', {a: (ARMS[a], clean) for a in ('T21', 'T21S')}, v2key)
OUT['v2-ref'] = summarize('t21-heldout-v2', {'T2': (ARMS['T2'], clean), 'SPG': (ARMS['SPG'], spg_clean)}, v2key)
OUT['v2-spg-subset'] = summarize('t21-heldout-v2-clean', {a: (ARMS[a], spg_clean) for a in ('T21', 'T21S')}, v2key)
Path(sys.argv[1]).write_text(json.dumps(OUT, indent=1)); print('wrote', sys.argv[1], {k: list(v.get('arms', {})) if isinstance(v, dict) and 'arms' in v else len(v) for k, v in OUT.items()})
