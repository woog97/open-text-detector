"""Space side: compare the three full-length MoE runs (final-moe on the T2.1-based final mix, moe-mixA, moe-mixB; stage-2
epoch 2, one seed each) on identical rows. Per set: recall at 1% FPR on that set's human-control sentences, sentence
AUROC vs controls, recall and human-control FPR at the paper threshold (1% FPR on the paper-edit set's dev human
sentences, per model), and document flag rates; per writer and per source on clean v2. v2 uses the clean-for-T2.1 ids
for every model (final-moe was scored on the full v2 file). Paired bootstrap (papers, 2,000 draws) of recall at 1% control
FPR, each pair. Usage: compare3.py OUT.json"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
MG = Path('/tmp/mixgrid-20261007/sweeps'); T2 = Path('/tmp/pangram-t2-20261006'); CK = 'stage2-epoch2'; rng = np.random.default_rng(7)
MODELS = {'T21': MG / 'final-moe', 'A': MG / 'moe-mixA', 'B': MG / 'moe-mixB'}
SW = [json.loads(l) for l in gzip.open(T2 / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt')]
V2 = {r['id']: r for r in map(json.loads, gzip.open(T2 / 'inputs/t21-heldout-score-rows-v2.jsonl.gz', 'rt'))}
CLEAN = {i for i, r in V2.items() if str(r.get('t21_may_have_seen_paper')) != 'True'}
SETS = [('v2', {'T21': 't21-heldout-v2', 'A': 't21-heldout-v2-clean', 'B': 't21-heldout-v2-clean'}), ('heldout', 'heldout-writers'), ('cross', 'cross-model-s500'),
        ('humanizer', 'fable-humanizer'), ('cowrite', 'fable-cowrite'), ('polish', 'fable-polish')]

def paper_thr(d):
    z = np.load(d / 'eval' / f'{CK}-sentences.npz')
    neg = np.concatenate([z[str(i)][z[str(i)][:, 1] == 0, 0] for i, r in enumerate(SW) if r['split'] == 'dev' and r['slot'] in ('edit_context', 'human_untouched', 'human_paper')])
    return float(np.sort(neg)[::-1][int(np.floor(.01 * len(neg)))])
def load(d, name):
    z = np.load(d / 'eval' / f'{name}-{CK}-sentences.npz'); rows = json.loads((d / 'eval' / f'{name}-{CK}-rows.json').read_text())
    return {r['id']: (r, z[str(i)]) for i, r in enumerate(rows)}
def auroc(pos, neg):
    s = np.r_[pos, neg]; o = np.argsort(s, kind='mergesort'); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
def paper(i):
    p = i.split('/'); return p[1] if p[0] in ('polish', 'humanizer', 'cowrite') and len(p) > 1 else p[0]

out = {'paper_threshold': {m: paper_thr(d) for m, d in MODELS.items()}, 'sets': {}, 'sweep': {}}
for m, d in MODELS.items():
    out['sweep'][m] = {e: json.loads((d / 'eval' / f'stage2-epoch{e}.json').read_text()) for e in (0, 1, 2) if (d / 'eval' / f'stage2-epoch{e}.json').exists()}
    for e, j in out['sweep'][m].items():
        out['sweep'][m][e] = {h: {k: (j[h][k] if not isinstance(j[h][k], dict) else {kk: vv for kk, vv in j[h][k].items()}) for k in j[h]} for h in ('dev', 'test')}
    sel = d / 'stage2-selection.json'; out['sweep'][m]['selected'] = json.loads(sel.read_text()).get('checkpoint') if sel.exists() else None
for key, names in SETS:
    data = {m: load(d, names[m] if isinstance(names, dict) else names) for m, d in MODELS.items()}
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    if key == 'v2': ids = [i for i in ids if i in CLEAN]
    res = {'n_rows': len(ids), 'models': {}}
    per_paper = {}
    for m, dat in data.items():
        thr_p = out['paper_threshold'][m]
        ctrl = np.concatenate([dat[i][1][dat[i][1][:, 1] == 0, 0] for i in ids if dat[i][0]['slot'] == 'human_controls'])
        thr_c = float(np.sort(ctrl)[::-1][int(np.floor(.01 * len(ctrl)))])
        g = defaultdict(list)
        for i in ids:
            r, x = dat[i]
            if r['slot'] != 'mixed': continue
            pos = x[x[:, 1] == 1, 0]
            for k in ['all', 'w:' + r['writer']] + (['src:' + V2[i]['source']] if key == 'v2' and i in V2 else []): g[k].append(pos)
        g = {k: np.concatenate(v) for k, v in g.items()}
        fp_src = defaultdict(list)
        if key == 'v2':
            for i in ids:
                r, x = dat[i]
                if r['slot'] == 'human_controls': fp_src['src:' + V2[i]['source']].append(x[x[:, 1] == 0, 0])
        docflag = lambda slot, t: float(np.mean([len(dat[i][1]) > 0 and dat[i][1][:, 0].max() > t for i in ids if dat[i][0]['slot'] == slot]))
        res['models'][m] = {'thr_controls_1pct': thr_c, 'thr_paper': thr_p,
                            'groups': {k: {'n': int(len(v)), 'r_ctrl1': float(np.mean(v > thr_c)) if len(v) else None, 'auroc': auroc(v, ctrl) if len(v) >= 5 else None,
                                           'r_paper': float(np.mean(v > thr_p)) if len(v) else None} for k, v in g.items()},
                            'ctrl_fpr_paper': float(np.mean(ctrl > thr_p)), 'n_ctrl_sent': int(len(ctrl)),
                            'ctrl_fpr_paper_by_src': {k: float(np.mean(np.concatenate(v) > thr_p)) for k, v in fp_src.items()},
                            'docflag_ai_paper': docflag('mixed', thr_p), 'docflag_ctrl_paper': docflag('human_controls', thr_p)}
        hits = defaultdict(lambda: [0, 0])
        for i in ids:
            r, x = dat[i]
            if r['slot'] == 'mixed': pos = x[x[:, 1] == 1, 0]; h = hits[paper(i)]; h[0] += int((pos > thr_c).sum()); h[1] += len(pos)
        per_paper[m] = hits
    papers = sorted(set().union(*[set(v) for v in per_paper.values()]))
    W = rng.multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=2000).T
    vec = {m: (np.array([per_paper[m][p][0] for p in papers], float), np.array([per_paper[m][p][1] for p in papers], float)) for m in MODELS}
    res['paired_r_ctrl1'] = {}
    for a, b in (('A', 'T21'), ('B', 'T21'), ('A', 'B')):
        ra = (vec[a][0] @ W) / np.maximum(vec[a][1] @ W, 1); rb = (vec[b][0] @ W) / np.maximum(vec[b][1] @ W, 1); dlt = ra - rb
        res['paired_r_ctrl1'][f'{a}-{b}'] = {'diff': float(vec[a][0].sum() / vec[a][1].sum() - vec[b][0].sum() / vec[b][1].sum()), 'ci95': [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]}
    out['sets'][key] = res
Path(sys.argv[1]).write_text(json.dumps(out, indent=1)); print('wrote', {k: v['n_rows'] for k, v in out['sets'].items()}, out['paper_threshold'])
