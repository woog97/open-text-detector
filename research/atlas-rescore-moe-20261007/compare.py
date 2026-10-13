"""Old Atlas 4B vs final MoE: per-paper sentence flag shares by venue/year (runs on the Space).

Old: live Atlas scorer outputs (classifications/<set>/qwen35-4b-fast10/sentences), at the live threshold 0.0283 and at
the pooled Oct 6 recalibration 0.0248. New: MoE token shards -> sentences with the same rule, at the MoE 1%-FPR
threshold (calibration-moe.json). Human 2019-2022 rows come from the two calibration analyses.
Usage: compare.py yd|atlas|all OUT.json
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

C = Path('/data/workspace/classifications'); D = Path('/data/workspace/atlas-rescore-moe-20261007')
OLD_T, OLD_T2 = 0.0283355712890625, 0.024772189557552338
NEW_T = json.loads((D / 'calibration-moe.json').read_text())['variants']['kept+old (as Atlas calibration)']['thresholds'][1]['threshold']
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)


def sentence_scores(text, off, prob):
    off = np.asarray(off).reshape(-1, 2); prob = np.asarray(prob, dtype=np.float64)
    valid = np.array([bool(text[a:b].strip()) for a, b in off], dtype=np.float64)
    cs = np.r_[0, np.cumsum(prob * valid)]; cv = np.r_[0, np.cumsum(valid)]; out = []
    for m in SENT.finditer(text):
        if not m.group().strip(): continue
        a, b = m.start(), m.end(); lo = np.searchsorted(off[:, 1], a, side='right'); hi = np.searchsorted(off[:, 0], b, side='left'); n = cv[hi] - cv[lo]
        if hi > lo and n > 0: out.append((m.start(), m.end(), (cs[hi] - cs[lo]) / n))
    return out


def old_scores(d):
    out = {}
    for f in sorted((d / 'sentences').glob('sentences-*.npz')):
        z = np.load(f)
        for k in z.files:
            pid, kind = k.rsplit('/', 1)
            if kind == 'scores': out[pid] = z[k].astype(float)
    return out


def new_scores(dirs, texts, keep_spans=False):
    out = {}; spans = {}
    for d in dirs:
        for f in sorted((d / 'tokens').glob('tokens-*.npz')):
            z = np.load(f); ids = sorted({k.rsplit('/', 1)[0] for k in z.files})
            for pid in ids:
                if pid in out or pid not in texts: continue
                s = sentence_scores(texts[pid], z[pid + '/offsets'], z[pid + '/probs'])
                out[pid] = np.array([x[2] for x in s]);
                if keep_spans: spans[pid] = s
    return out, spans


def stats(shares):
    v = np.asarray(shares, float)
    if not len(v): return None
    return {'papers': int(len(v)), 'mean': float(v.mean()), 'p25': float(np.quantile(v, .25)), 'p50': float(np.median(v)), 'p75': float(np.quantile(v, .75)),
            'over_5pct': float((v > .05).mean()), 'over_25pct': float((v > .25).mean()), 'over_50pct': float((v > .5).mean())}


def compare_set(name, text_dirs, old_dir, new_dirs, group_of, keep_spans=False):
    meta = {}; texts = {}
    for d in text_dirs:
        for f in sorted((d / 'data').glob('*.parquet')):
            cols = [c for c in ('id', 'text', 'conference', 'year', 'source', 'title') if c in pq.read_schema(f).names]
            for r in pq.read_table(f, columns=cols).to_pylist(): texts[r['id']] = r.pop('text'); meta[r['id']] = r
    old = old_scores(old_dir); new, spans = new_scores(new_dirs, texts, keep_spans)
    both = sorted(set(old) & set(new))
    groups = defaultdict(lambda: defaultdict(list)); per = {}
    for pid in both:
        o, n = old[pid], new[pid]
        if not len(o) or not len(n): continue
        g = group_of(meta[pid]); x = {'old_live': float((o > OLD_T).mean()), 'old_recal': float((o > OLD_T2).mean()), 'new': float((n > NEW_T).mean())}
        per[pid] = x
        for k, v in x.items(): groups[g][k].append(v); groups['ALL ' + name][k].append(v)
    res = {g: {k: stats(v) for k, v in d.items()} for g, d in sorted(groups.items())}
    o = np.array([per[p]['old_live'] for p in per]); n = np.array([per[p]['new'] for p in per])
    shift = {'papers': len(per), 'corr_per_paper_share': float(np.corrcoef(o, n)[0, 1]) if len(per) > 2 else None,
             'rank_corr': float(np.corrcoef(o.argsort().argsort(), n.argsort().argsort())[0, 1]) if len(per) > 2 else None,
             'scored_old_only': len(set(old) - set(new)), 'scored_new_only': len(set(new) - set(old))}
    spot = []
    if keep_spans and per:
        diff = sorted(per, key=lambda p: per[p]['new'] - per[p]['old_live'])
        rng = np.random.default_rng(7); pick = diff[:3] + diff[-3:] + list(rng.choice(sorted(per), 4, replace=False))
        for p in pick:
            s = sorted(spans[p], key=lambda x: -x[2])[:2]; t = texts[p]
            spot.append({'id': p, 'title': (meta[p].get('title') or '')[:120], **{k: round(v, 4) for k, v in per[p].items()},
                         'top_new_sentences': [{'score': round(float(sc), 4), 'text': t[a:b][:300]} for a, b, sc in s]})
    return {'groups': res, 'shift': shift, 'spot_checks': spot}


def main(which, out):
    res = {'thresholds': {'old_live': OLD_T, 'old_recal': OLD_T2, 'new': NEW_T}}
    if which in ('yd', 'all'):
        res['year_diagnostic'] = compare_set('year-diagnostic', [C / 'calibration-inputs/year-diagnostic-v1'], C / 'year-diagnostic-v1-20261006/qwen35-4b-fast10',
                                             [D / 'out-yd'], lambda m: f"{m.get('conference')} {m.get('year')} ({m.get('source')})")
    if which in ('atlas', 'all'):
        ds = Path('/data/workspace/datasets')
        res['atlas'] = compare_set('ICLR 2027 (Atlas)', [ds / 'iclr2027-clean-text-v2-20261005', ds / 'iclr2027-backfill-clean-text-v2-20261005'],
                                   C / 'iclr2027-clean-v2-20261005/qwen35-4b-fast10', [D / 'out-space', D / 'out-h200'], lambda m: 'iclr 2027', keep_spans=True)
    cal_new = json.loads((D / 'calibration-moe.json').read_text())['variants']['kept+old (as Atlas calibration)']['groups']
    cal_old = json.loads(Path('/data/workspace/human-calibration-20261006/results.json').read_text())['models']['atlas-fast10']['groups']
    res['human_calibration_by_year'] = [{'venue': n['venue'], 'year': n['year'], 'papers': n['papers'], 'new_mean': n['flagged_share_mean'], 'new_p50': n['flagged_share_p50'],
                                         **next(({'old_recal_mean': o['flagged_share_mean'], 'old_recal_p50': o['flagged_share_p50']} for o in cal_old
                                                 if o['venue'] == n['venue'] and o['year'] == n['year']), {})} for n in cal_new]
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != 'atlas'}, indent=1)[:6000])
    if 'atlas' in res: print(json.dumps(res['atlas']['groups'], indent=1), json.dumps(res['atlas']['shift']))


if __name__ == '__main__':
    main(*sys.argv[1:3])
