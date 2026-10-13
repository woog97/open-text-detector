import sys, numpy as np, pandas as pd
bench, root = sys.argv[1], sys.argv[2]
pairs = pd.read_parquet(f'{bench}/pairs.parquet'); sc = pd.read_parquet(f'{bench}/scores.parquet').drop_duplicates('paper_id').set_index('paper_id')
ours = pd.read_csv(f'{root}/result-prose.csv').set_index('paper_id')
h = pairs.drop_duplicates('human_paper_id').dropna(subset=['human_iclr_rating']).set_index('human_paper_id')
h = h.join(sc[['scislop_aggregate', 'scislop_xsec_ref', 'scislop_macro_redund', 'pangram']]).join(ours[['flag_bibex', 'doc']])
print('rated human papers:', len(h)); print(h.human_venue.value_counts().head(12).to_string()); print(h.H1_by.value_counts(dropna=False).to_string())
print('v1 year:', h.human_v1_year.value_counts(dropna=False).sort_index().to_string())
def sp(g, c):
    m = g[[c, 'human_iclr_rating']].dropna(); return round(m[c].rank().corr(m.human_iclr_rating.rank()), 3), len(m)
groups = {'all': h, 'iclr2026_pangram_filtered': h[h.H1_by == 'pangram'], 'v1<=2024': h[h.H1_by != 'pangram'], 'v1<=2022': h[h.human_v1_year <= 2022]}
for n, g in groups.items():
    print(n, len(g), 'rating median', g.human_iclr_rating.median(), {c: sp(g, c) for c in ['scislop_aggregate', 'scislop_xsec_ref', 'scislop_macro_redund', 'flag_bibex', 'pangram']})
print('accepted share by venue string:', (h.human_venue.str.contains('poster|oral|spotlight', case=False)).mean())
