import sys, re, glob, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[3]); from analyze_lib import sent_spans_scores, auroc
bench, root = sys.argv[1], sys.argv[2]
papers = pd.read_parquet(f'{bench}/papers.parquet').set_index('paper_id'); pairs = pd.read_parquet(f'{bench}/pairs.parquet')
hp = pairs.drop_duplicates('human_paper_id').set_index('human_paper_id')
old = set(hp.index[hp.human_v1_year <= 2022]); ai = set(papers.index[papers.label == 1])
out = {}
for view, col in [('prose', 'body_txt'), ('tex', 'body_tex')]:
    d = pd.read_csv(f'{root}/result-{view}.csv').set_index('paper_id')
    a = d.loc[list(ai)]; h = d.loc[list(old)]
    m = {}
    for s in ['doc', 'tokmean', 'flag_bibex']:
        t5 = np.quantile(h[s], .95, method='higher')
        m[s] = {'auroc_vs_pre2023': round(auroc(a[s], h[s]), 3), 'tpr_at_human95pct': round(float((a[s] > t5).mean()), 3)}
    # sentence-level: flag rate in pre-2023 human papers by placeholder presence
    stats = {'ph': [0, 0], 'clean': [0, 0]}; examples = []
    for f in sorted(glob.glob(f'{root}/out-scislop-{view}/tokens/*.npz')):
        z = np.load(f)
        for pid in {k.rsplit('/', 1)[0] for k in z.files} & old:
            text = papers.loc[pid, col]
            for (a0, b0), sc in sent_spans_scores(text, z[pid + '/offsets'], z[pid + '/probs'].astype(np.float32)):
                sent = text[a0:b0]; k = 'ph' if re.search(r'\[MATH\]|\bCITE\b|\bREF\b|\\cite|\\ref|\$', sent) else 'clean'
                stats[k][0] += 1; stats[k][1] += sc > 0.00885
                if k == 'clean' and sc > 0.00885 and len(examples) < 6 and len(sent) > 80: examples.append(sent[:220].replace('\n', ' '))
    m['pre2023_sentence_flag_rate'] = {k: {'sentences': v[0], 'flag_rate': round(v[1] / max(v[0], 1), 3)} for k, v in stats.items()}
    m['clean_flagged_examples'] = examples
    out[view] = m
print(json.dumps(out, indent=1))
json.dump(out, open(f'{root}/extra.json', 'w'), indent=1)
