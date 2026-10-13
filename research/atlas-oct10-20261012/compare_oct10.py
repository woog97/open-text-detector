"""Oct 7 vs Oct 10 paper-level flagged shares on the Atlas and on the human calibration papers (runs anywhere).

For each counting mode shown on the site (nobib, body; 1% and 0.1% FPR) and each model, with that model's own
thresholds: the human calibration papers' flagged shares (cutoff-explorer human-sentences.parquet, the 3 papers
dropped by recalibrate.py excluded, papers with >= 50 counted sentences, score >= cutoff), their median, mean and
99th percentile; and for the ICLR 2027 Atlas papers (the live flag index for Oct 7, build_oct10.py's flags for
Oct 10) the median share, the share of papers above the human 99th percentile, and the share with no flagged
sentence, overall and by primary area if a browse papers.json is given.
Usage: compare_oct10.py LIVE_FLAGS.json FLAGS_OCT10.json HUMAN_OCT7_DIR HUMAN_OCT10_DIR OUT.json [BROWSE_PAPERS.json]
"""
import json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

EXCLUDE = {'old:a46aaf09ea4a50245ec81d5b6727af529416788f303f8cf1155fafe13f047010',
           'old:857075500638db9ce6edcf32d990aa77a686bb4fbcd77126a8ed231cf52b78e4',
           'old:06ab694121b8f93ca2ef6aecaee2139d0ad417c4255a12a89930e0b1fb42cdbc'}
SEC = {'nobib': (0, 2), 'body': (0,), 'all': (0, 1, 2), 'noapp': (0, 1)}
MIN = 50


def human_shares(d, th):
    p = pq.read_table(Path(d) / 'human-papers.parquet').to_pylist(); s = pq.read_table(Path(d) / 'human-sentences.parquet').to_pydict()
    keep = np.array([x['key'] not in EXCLUDE for x in p]); paper, sec, sc = np.array(s['paper']), np.array(s['section']), np.array(s['score'])
    out = {}
    for m, t in th.items():
        k = np.isin(sec, SEC[m.removesuffix('0.1')]); tot = np.bincount(paper[k], minlength=len(p)); fl = np.bincount(paper[k & (sc >= t)], minlength=len(p))
        ok = keep & (tot >= MIN); out[m] = fl[ok] / tot[ok]
    return out, int(keep.sum())


def iclr_shares(flags, m, ids=None):
    r = {i: v['modes'][m] for i, v in flags['papers'].items() if (ids is None or i in ids) and v['modes'][m][1] >= MIN}
    return np.array([a / b for a, b in r.values()]), list(r)


def main(live, new, h7, h10, out, browse=None):
    live, new = json.loads(Path(live).read_text()), json.loads(Path(new).read_text()); res = {}
    modes = ['nobib0.1', 'body0.1', 'nobib', 'body', 'all0.1', 'all']
    for name, fl, hd in (('oct7', live, h7), ('oct10', new, h10)):
        th = {m: fl['thresholds'][m] for m in modes}; hs, n = human_shares(hd, th); res[name] = {'human_papers': n}
        for m in modes:
            h = hs[m]; p99 = float(np.quantile(h, .99)); x, _ = iclr_shares(fl, m)
            res[name][m] = {'threshold': th[m], 'human': {'papers': len(h), 'median': float(np.median(h)), 'mean': float(h.mean()), 'p99': p99,
                                                           'share_zero': float((h == 0).mean())},
                            'iclr2027': {'papers': len(x), 'median': float(np.median(x)), 'mean': float(x.mean()), 'over_human_p99': float((x > p99).mean()),
                                         'share_zero': float((x == 0).mean()), 'over_50pct': float((x >= .5).mean())}}
    if browse:
        rows = json.loads(Path(browse).read_text())['items']; areas = {}
        for r in rows:
            if r.get('collection') == 'iclr/2027' and r.get('primary_area'): areas.setdefault(r['primary_area'], set()).add(r['id'])
        res['by_area_nobib0.1'] = {}
        for a, ids in sorted(areas.items(), key=lambda kv: -len(kv[1])):
            row = {}
            for name, fl in (('oct7', live), ('oct10', new)):
                x, _ = iclr_shares(fl, 'nobib0.1', ids); p99 = res[name]['nobib0.1']['human']['p99']
                row[name] = {'papers': len(x), 'median': float(np.median(x)) if len(x) else None, 'over_human_p99': float((x > p99).mean()) if len(x) else None}
            res['by_area_nobib0.1'][a] = row
    a, b = iclr_shares(live, 'nobib0.1'); c, d = iclr_shares(new, 'nobib0.1')
    common = sorted(set(b) & set(d)); va = dict(zip(b, a)); vc = dict(zip(d, c))
    res['paired_nobib0.1'] = {'papers': len(common), 'spearman': float(np.corrcoef(np.argsort(np.argsort([va[i] for i in common])), np.argsort(np.argsort([vc[i] for i in common])))[0, 1]),
                              'mean_diff_oct10_minus_oct7': float(np.mean([vc[i] - va[i] for i in common]))}
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
