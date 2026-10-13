import sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[3]); from analyze_lib import auroc
bench, root = sys.argv[1], sys.argv[2]
pairs = pd.read_parquet(f'{bench}/pairs.parquet'); sc = pd.read_parquet(f'{bench}/scores.parquet').drop_duplicates('paper_id').set_index('paper_id')
ours = pd.read_csv(f'{root}/result-prose.csv').set_index('paper_id')[['tokmean', 'flag_bibex', 'doc']]
cols = ['scislop_aggregate', 'scislop_macro_redund', 'scislop_xsec_ref', 'scislop_argument_graph', 'scislop_citation', 'scislop_fig_exposition', 'scislop_evidence_gap', 'binoculars', 'pangram', 'cyclereviewer_overall', 'ai_scientist_overall']
d = sc[cols].join(ours)
def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y)); return float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())), int(m.sum())
out = {}
# 1) Agents4Science AI papers: accepted vs rejected, and mean reviewer score
a = pairs[pairs.source == 'agents4science'].copy()
def mean_score(js):
    try:
        v = json.loads(js); v = list(v.values()) if isinstance(v, dict) else v
        nums = []
        for x in v:
            if isinstance(x, dict): x = x.get('overall', x.get('rating', x.get('score')))
            try: nums.append(float(str(x).split(':')[0]))
            except Exception: pass
        return float(np.mean(nums)) if nums else np.nan
    except Exception: return np.nan
a['rev'] = a.a4s_reviewer_scores.map(mean_score)
print('example reviewer field:', a.a4s_reviewer_scores.dropna().iloc[0][:200])
A = d.reindex(a.ai_paper_id).assign(acc=(a.a4s_decision.str.lower().str.startswith('accept')).to_numpy(), rev=a.rev.to_numpy())
res = {}
for c in cols + ['tokmean', 'flag_bibex', 'doc']:
    x = A[c].to_numpy(float); ok = ~np.isnan(x)
    acc = A.acc.to_numpy()[ok]; x = x[ok]
    # AUROC that REJECTED papers score higher (higher slop/AI = worse)
    res[c] = {'auroc_rejected_higher': round(auroc(x[~acc], x[acc]), 3), 'n_acc': int(acc.sum()), 'n_rej': int((~acc).sum()),
              'spearman_vs_reviewer_mean': [round(v, 3) if isinstance(v, float) else v for v in spearman(A[c].to_numpy(float), A.rev.to_numpy(float))]}
out['a4s_ai_accept_vs_reject'] = res
# 2) Human anchors: Spearman with mean ICLR rating
h = pairs.drop_duplicates('human_paper_id').dropna(subset=['human_iclr_rating'])
H = d.reindex(h.human_paper_id); r = h.human_iclr_rating.to_numpy(float)
out['human_vs_iclr_rating'] = {c: [round(v, 3) if isinstance(v, float) else v for v in spearman(H[c].to_numpy(float), r)] for c in cols + ['tokmean', 'flag_bibex', 'doc']}
out['human_rating_range'] = [float(np.nanmin(r)), float(np.nanmedian(r)), float(np.nanmax(r))]
json.dump(out, open(f'{root}/quality.json', 'w'), indent=1); print(json.dumps(out, indent=1))
