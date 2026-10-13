"""Space side (CPU): compare the final runs with T2.1 / T2 / SPG (4B, 3 seeds each, stage2-epoch2), the way ab_t21.py reports.

Reads final-run scores from /data/workspace/final-20261007/eval/<tag>/ (uploaded by `eval_h200.py collect`) and the
T2 workspace scores from /data/workspace/pangram-t2-20261006/sweeps/<tag>/eval/. Writes eval/report.json and report.md.

Per evaluation set: recall of AI sentences at the in-domain threshold (1% FPR on the sweep-eval dev human sentences,
per checkpoint), overall / small edits / paragraph edits / per writer; human-control sentence FPR at that threshold;
AUROC of AI sentences vs human-control sentences. Sweep eval: dev/test small-edit recall at each split's own 1% FPR, AUROC.
T2.1 held-out v2: compared on the rows T2.1 never saw (`t21-heldout-v2-clean` ids); the final runs never saw any v2 paper.
Final runs are one seed; reference arms are 3-seed means (seed spread about 1 point on small-edit recall at 1% FPR),
except wide1-heldout, where only T2.1 seed 1 was scored (eval/ref-q4b-T21-s1).
"""
import gzip, json
from pathlib import Path
import numpy as np

F = Path('/data/workspace/final-20261007/eval'); T2 = Path('/data/workspace/pangram-t2-20261006/sweeps')
SW_PATH = next(p for p in [T2 / 'sweep-eval-rows.jsonl.gz', Path('/tmp/pangram-t2-20261006/sweeps/sweep-eval-rows.jsonl.gz')] if p.exists())
SW = [json.loads(l) for l in gzip.open(SW_PATH, 'rt')]
FINAL = ['final-eg2', 'final-qwen35-4b', 'final-moe']
REF = {a: [f'q4b-{a}-s{s}' for s in (1, 2, 3)] for a in ('T21', 'T2', 'SPG')}
SMALL = {'rewrite_one', 'rewrite_two', 'insert_one', 'split_one', 'sentence', 'two_sentence'}
SETS = ['heldout-writers', 't21-heldout-v2', 'cross-model-s500', 'fable-polish', 'fable-humanizer', 'fable-cowrite', 'wide1-heldout']


def auroc(pos, neg):
    if len(pos) < 5 or len(neg) < 5: return None
    s = np.r_[pos, neg]; o = np.argsort(s, kind='mergesort'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def thr1(neg): return float(np.sort(neg)[::-1][int(np.floor(.01 * len(neg)))])


def sweep_sents(ev, ck):
    z = np.load(ev / f'{ck}-sentences.npz'); return [z[str(i)] for i in range(len(SW))]


def human_neg(sents, split):
    return np.concatenate([x[x[:, 1] == 0, 0] for r, x in zip(SW, sents) if r['split'] == split and r['slot'] in ('edit_context', 'human_untouched', 'human_paper')])


def sweep_metrics(ev, ck):
    x = sweep_sents(ev, ck); out = {}
    for split in ('dev', 'test'):
        neg = human_neg(x, split); th = thr1(neg)
        pos = np.concatenate([s[s[:, 1] == 1, 0] for r, s in zip(SW, x) if r['split'] == split and r['slot'] == 'edit_context' and r['condition'] in ('sentence', 'two_sentence')])
        para = np.concatenate([s[s[:, 1] == 1, 0] for r, s in zip(SW, x) if r['split'] == split and r['slot'] == 'edit_context' and r['condition'] in ('paragraph_concise', 'paragraph_v3')])
        out[split] = {'small_recall_1pct': float(np.mean(pos > th)), 'small_auroc': auroc(pos, neg), 'para_recall_1pct': float(np.mean(para > th)), 'n_small': len(pos)}
    return out, thr1(human_neg(x, 'dev'))


def set_metrics(ev, name, ck, thr, ids=None):
    f = ev / f'{name}-{ck}-sentences.npz'
    if not f.exists(): return None
    z = np.load(f); rows = json.loads((ev / f'{name}-{ck}-rows.json').read_text())
    keep = [(r, z[str(i)]) for i, r in enumerate(rows) if ids is None or r['id'] in ids]
    neg = np.concatenate([x[x[:, 1] == 0, 0] for r, x in keep if r['slot'] == 'human_controls'] + [np.zeros(0)])
    groups = {}
    for r, x in keep:
        if r['slot'] != 'mixed': continue
        pos = x[x[:, 1] == 1, 0]; c = str(r.get('condition'))
        for g in ('all', 'w:' + str(r.get('writer')), 'small' if c in SMALL else ('para' if c.startswith(('para', 'paragraph')) else 'other')):
            groups.setdefault(g, []).append(pos)
    res = {'n_rows': len(keep), 'control_fpr': float(np.mean(neg > thr)) if len(neg) else None, 'n_control_sentences': len(neg)}
    for g, v in sorted(groups.items()):
        pos = np.concatenate(v)
        if len(pos): res[g] = {'n': int(len(pos)), 'recall': float(np.mean(pos > thr)), 'auroc': auroc(pos, neg)}
    return res


def mean_of(results):
    results = [r for r in results if r]
    if not results: return None
    out = {}
    for k in results[0]:
        vals = [r[k] for r in results if k in r]
        if isinstance(vals[0], dict):
            out[k] = {kk: (float(np.mean([v[kk] for v in vals if v.get(kk) is not None])) if any(v.get(kk) is not None for v in vals) else None) for kk in vals[0]}
        elif isinstance(vals[0], (int, float)) and vals[0] is not None:
            out[k] = float(np.mean([v for v in vals if v is not None]))
        else:
            out[k] = vals[0]
    out['n_seeds'] = len(results)
    return out


def main():
    v2clean = None
    for t in REF['T21']:
        p = T2 / t / 'eval' / 't21-heldout-v2-clean-stage2-epoch2-rows.json'
        if p.exists(): v2clean = {r['id'] for r in json.loads(p.read_text())}; break
    rep = {'final': {}, 'reference': {}}
    for tag in FINAL:
        ev = F / tag
        if not (ev / 'summary.json').exists(): continue
        summ = json.loads((ev / 'summary.json').read_text()); rep['final'][tag] = {'selected': summ['selected']}
        for ck in summ['checkpoints']:
            sm, th = sweep_metrics(ev, ck); r = {'sweep': sm, 'threshold_in_domain': th}
            for name in SETS:
                r[name] = set_metrics(ev, name, ck, th, v2clean if name == 't21-heldout-v2' else None)
            rep['final'][tag][ck] = r
    for arm, tags in REF.items():
        per = []
        for t in tags:
            ev = T2 / t / 'eval'
            if not (ev / 'stage2-epoch2-sentences.npz').exists(): continue
            sm, th = sweep_metrics(ev, 'stage2-epoch2'); r = {'sweep': sm}
            for name in SETS:
                n = 't21-heldout-v2-clean' if (name == 't21-heldout-v2' and arm == 'T21') else name
                ev2 = ev if (ev / f'{n}-stage2-epoch2-sentences.npz').exists() else F / f'ref-{t}'  # wide1: scored later on the H200 (seed 1 only)
                r[name] = set_metrics(ev2, n, 'stage2-epoch2', th, v2clean if name == 't21-heldout-v2' else None)
            per.append(r)
        if per:
            rep['reference'][arm] = {'sweep': {s: mean_of([p['sweep'][s] for p in per]) for s in ('dev', 'test')},
                                     **{name: mean_of([p[name] for p in per]) for name in SETS}}
    (F / 'report.json').write_text(json.dumps(rep, indent=1))
    # Markdown: headline table on the selected checkpoint of each final run vs the reference arms.
    cols = [(tag, rep['final'][tag][rep['final'][tag]['selected']]) for tag in rep['final']] + [(a + ' (3-seed mean)', v) for a, v in rep['reference'].items()]
    def g(v, *path):
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
        return '—' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
    lines = ['| Metric | ' + ' | '.join(c for c, _ in cols) + ' |', '|---|' + '---|' * len(cols)]
    rowspec = [('Sweep test: small-edit recall @1% FPR', ('sweep', 'test', 'small_recall_1pct')), ('Sweep test: small-edit AUROC', ('sweep', 'test', 'small_auroc')),
               ('Sweep dev: small-edit recall @1% FPR', ('sweep', 'dev', 'small_recall_1pct')), ('Sweep test: paragraph recall @1% FPR', ('sweep', 'test', 'para_recall_1pct'))]
    for name in SETS:
        rowspec += [(f'{name}: small-edit recall (in-domain thr)', (name, 'small', 'recall')), (f'{name}: all AI recall (in-domain thr)', (name, 'all', 'recall')),
                    (f'{name}: AUROC vs controls', (name, 'all', 'auroc')), (f'{name}: control FPR', (name, 'control_fpr'))]
    for label, path in rowspec:
        lines.append(f'| {label} | ' + ' | '.join(g(v, *path) for _, v in cols) + ' |')
    wl = ['', '**wide1 held-out by writer (selected checkpoints):**', '', '| Writer group | ' + ' | '.join(t for t in rep['final']) + ' |', '|---|' + '---|' * len(rep['final'])]
    keys = sorted({k for t in rep['final'] for k in (rep['final'][t][rep['final'][t]['selected']].get('wide1-heldout') or {}) if isinstance((rep['final'][t][rep['final'][t]['selected']].get('wide1-heldout') or {}).get(k), dict)})
    for k in keys:
        wl.append(f'| {k} | ' + ' | '.join(g(rep['final'][t][rep['final'][t]['selected']], 'wide1-heldout', k, 'recall') + ' / ' + g(rep['final'][t][rep['final'][t]['selected']], 'wide1-heldout', k, 'auroc') for t in rep['final']) + ' |')
    md = '# Final runs vs T2.1 / T2 / SPG\n\nRecall at the in-domain threshold (1% FPR on sweep-eval dev human sentences); wide1 cells are recall / AUROC.\n\n' + '\n'.join(lines + wl) + '\n'
    (F / 'report.md').write_text(md); print(md)


if __name__ == '__main__':
    main()
