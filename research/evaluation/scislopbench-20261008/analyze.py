"""Score SciSlopBench papers with final-moe outputs (score_moe.py format) and compare with SciSlop/Pangram.

usage: analyze.py BENCH_DATA_DIR OUT_ROOT RESULT.json
  BENCH_DATA_DIR: pairs/papers/scores parquet from the release; OUT_ROOT: holds out-scislop-prose/, out-scislop-tex/
Paper scores: document head prob, mean token prob, and the Atlas sentence rule (regex sentences, mean token prob)
-> share of sentences above the MoE 1%-FPR sentence thresholds (0.00885 without bibliographies; 0.00584 with).
"""
import glob, json, re, sys
import numpy as np, pandas as pd

bench, root, out_path = sys.argv[1:4]
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
THR = {'bibex': 0.00885, 'withbib': 0.00584}


def sent_scores(text, off, prob):
    off = np.asarray(off).reshape(-1, 2); prob = np.asarray(prob, np.float64)
    valid = np.array([bool(text[a:b].strip()) for a, b in off], np.float64)
    cs = np.r_[0, np.cumsum(prob * valid)]; cv = np.r_[0, np.cumsum(valid)]; sc = []
    for m in SENT.finditer(text):
        if not m.group().strip(): continue
        a, b = m.start(), m.end()
        lo = np.searchsorted(off[:, 1], a, side='right'); hi = np.searchsorted(off[:, 0], b, side='left')
        n = cv[hi] - cv[lo]
        if hi > lo and n > 0: sc.append((cs[hi] - cs[lo]) / n)
    return np.array(sc)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.r_[pos, neg]; r = pd.Series(allv).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at(pos, neg, fpr):
    t = np.quantile(neg, 1 - fpr, method='higher')  # human papers above t <= fpr
    return float((np.asarray(pos) > t).mean())


papers = pd.read_parquet(f'{bench}/papers.parquet')
pairs = pd.read_parquet(f'{bench}/pairs.parquet')
ext = pd.read_parquet(f'{bench}/scores.parquet').drop_duplicates('paper_id').set_index('paper_id')
views = {}
for view, col in [('prose', 'body_txt'), ('tex', 'body_tex')]:
    text = dict(zip(papers.paper_id, papers[col].fillna('')))
    rows = []
    summ = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{root}/out-scislop-{view}/data/*.parquet'))]).set_index('id')
    for f in sorted(glob.glob(f'{root}/out-scislop-{view}/tokens/*.npz')):
        z = np.load(f); ids = {k.rsplit('/', 1)[0] for k in z.files}
        for pid in ids:
            sc = sent_scores(text[pid], z[pid + '/offsets'], z[pid + '/probs'].astype(np.float32))
            rows.append({'paper_id': pid, 'sentences': len(sc), 'doc': summ.loc[pid, 'document_prob'],
                         'tokmean': summ.loc[pid, 'token_prob_mean'],
                         **{f'flag_{k}': float((sc > v).mean()) if len(sc) else np.nan for k, v in THR.items()}})
    views[view] = pd.DataFrame(rows).set_index('paper_id')
    assert len(views[view]) == len(papers), (view, len(views[view]))

meta = papers.set_index('paper_id')[['label', 'source']]
ai_src = pairs.set_index('ai_paper_id')['ai_tex_source']
res = {'n_papers': len(papers), 'n_pairs': len(pairs), 'thresholds': THR, 'views': {}}
rng = np.random.default_rng(0)
for view, d in views.items():
    d = d.join(meta)
    d['ext_scislop'] = ext['scislop_aggregate']; d['ext_pangram'] = ext['pangram']; d['ext_binoculars'] = ext['binoculars']
    V = {}
    for score in ['doc', 'tokmean', 'flag_bibex', 'flag_withbib', 'ext_scislop', 'ext_pangram']:
        s = d[score]
        sub = {}
        for name, aimask in [('all', pairs.index == pairs.index), ('fars', pairs.source == 'fars'),
                             ('a4s_shipped_tex', (pairs.source == 'agents4science') & (pairs.ai_tex_source == 'shipped_tex')),
                             ('a4s_pdf_rebuilt', (pairs.source == 'agents4science') & (pairs.ai_tex_source == 'rebuilt_from_pdf'))]:
            pp = pairs[aimask]
            a = s.reindex(pp.ai_paper_id).to_numpy(); h = s.reindex(pp.human_paper_id).to_numpy()
            ok = ~np.isnan(a) & ~np.isnan(h)
            a, h = a[ok], h[ok]
            if not len(a): continue
            hu = s.reindex(pp.human_paper_id.unique()).dropna().to_numpy()
            pa = float(((a > h) + 0.5 * (a == h)).mean())
            boots = []
            for _ in range(1000):
                i = rng.integers(0, len(a), len(a)); boots.append(auroc(a[i], h[i]))
            sub[name] = {'pairs': int(len(a)), 'pair_acc': pa, 'auroc': auroc(a, hu), 'auroc_ci95': [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                         'tpr_at_5fpr': tpr_at(a, hu, .05), 'tpr_at_1fpr': tpr_at(a, hu, .01)}
        V[score] = sub
    desc = {}
    for lab in (0, 1):
        x = d[d.label == lab]
        desc['ai' if lab else 'human'] = {'papers': int(len(x)), 'sentences_median': float(x.sentences.median()),
            'flag_bibex_median': float(x.flag_bibex.median()), 'flag_bibex_mean': float(x.flag_bibex.mean()),
            'flag_bibex_over50': float((x.flag_bibex > .5).mean()), 'flag_bibex_over10': float((x.flag_bibex > .1).mean()),
            'flag_bibex_under1': float((x.flag_bibex < .01).mean()),
            'doc_median': float(x.doc.median()), 'doc_over_0_5': float((x.doc > .5).mean())}
    # human papers: pre-2023 arXiv vs ICLR 2026 (Pangram-filtered)
    hp = pairs.drop_duplicates('human_paper_id').set_index('human_paper_id')
    hum = d[d.label == 0].join(hp[['H1_by', 'human_v1_year']])
    groups = {'v1_le_2022': hum[hum.human_v1_year <= 2022], 'v1_2023_2024': hum[(hum.human_v1_year >= 2023) & (hum.human_v1_year <= 2024)],
              'pangram_filtered': hum[hum.H1_by == 'pangram']}
    desc['human_groups'] = {k: {'papers': int(len(g)), 'flag_bibex_median': float(g.flag_bibex.median()), 'flag_bibex_mean': float(g.flag_bibex.mean()),
                                 'flag_bibex_over10': float((g.flag_bibex > .1).mean()), 'doc_over_0_5': float((g.doc > .5).mean())} for k, g in groups.items()}
    # AI papers by self-reported writing involvement (Agents4Science)
    inv = pairs.set_index('ai_paper_id')['a4s_ai_involvement'].dropna()
    w = {}
    for pid, js in inv.items():
        try: w[pid] = json.loads(js).get('writing')
        except Exception: pass
    ai = d[d.label == 1].copy(); ai['writing'] = pd.Series(w)
    desc['a4s_by_writing'] = {str(k): {'papers': int(len(g)), 'flag_bibex_median': float(g.flag_bibex.median()), 'doc_median': float(g.doc.median())}
                              for k, g in ai[ai.source == 'agents4science'].groupby('writing')}
    res['views'][view] = {'metrics': V, 'describe': desc}
    d.to_csv(out_path.replace('.json', f'-{view}.csv'))
json.dump(res, open(out_path, 'w'), indent=1)
print(json.dumps({v: {s: {k: round(m['all'][k], 3) for k in ('pair_acc', 'auroc', 'tpr_at_5fpr', 'tpr_at_1fpr')} for s, m in x['metrics'].items()} for v, x in res['views'].items()}, indent=1))
print(json.dumps({v: x['describe'] for v, x in res['views'].items()}, indent=1))
