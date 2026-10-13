"""1%-FPR sentence threshold for the final MoE on the human calibration set (runs on the Space after score_moe.py).

Same rules as research/human-calibration-20261006/analyze_calibration.py (functions copied so this runs standalone):
sentence score = mean AI probability of non-whitespace tokens in each regex sentence (sentence_scores_space.py);
threshold_at = smallest threshold whose flag rate (score > t) on the human sentences is <= target; intervals resample
papers (flags cluster within papers); held-out check = threshold from half A (cal-split-v1 hash), flag rate on half B.
Papers: the 2,188 Oct 6 calibration papers kept by overlap.json (training-overlap screen) pooled with the 215 Oct 5
papers (prefixed old:), as for the Atlas model. Also reports the 2,299-paper version without the overlap screen.
Usage: cal_analyze.py OUT_CAL_DIR CAL_TEXT_DIR OLD_TEXT_DIR HUMANCAL_DIR OUT.json
"""
import hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
TARGETS = [0.001, 0.01, 0.05]; B = 1000
rng = np.random.default_rng(20261007)


def sentence_scores(text, off, prob):
    off = np.asarray(off).reshape(-1, 2); prob = np.asarray(prob, dtype=np.float64)
    valid = np.array([bool(text[a:b].strip()) for a, b in off], dtype=np.float64)
    cs = np.r_[0, np.cumsum(prob * valid)]; cv = np.r_[0, np.cumsum(valid)]; out = []
    for m in SENT.finditer(text):
        if not m.group().strip(): continue
        a, b = m.start(), m.end(); lo = np.searchsorted(off[:, 1], a, side='right'); hi = np.searchsorted(off[:, 0], b, side='left'); n = cv[hi] - cv[lo]
        if hi > lo and n > 0: out.append((cs[hi] - cs[lo]) / n)
    return np.array(out)


def threshold_at(neg, fpr=0.01):
    negs = np.sort(np.asarray(neg, float))[::-1]; return float(negs[int(np.floor(fpr * len(negs)))])


def half(pid): return int(hashlib.sha256(('cal-split-v1:' + pid).encode()).hexdigest(), 16) % 2


def weighted_threshold(scores, owner, weights, fpr):
    order = np.argsort(-scores, kind='mergesort'); w = weights[owner[order]]; cum = np.cumsum(w); total = cum[-1]
    i = np.searchsorted(cum, np.floor(fpr * total), side='right'); return float(scores[order][min(i, len(order) - 1)])


def paper_boot_rate(flags, counts, n=B):
    idx = rng.integers(0, len(flags), size=(n, len(flags))); rates = flags[idx].sum(1) / np.maximum(counts[idx].sum(1), 1)
    return [float(np.quantile(rates, .025)), float(np.quantile(rates, .975))]


def sentence_block(cal):
    pids = sorted(cal); scores = np.concatenate([cal[p] for p in pids])
    owner = np.concatenate([np.full(len(cal[p]), i) for i, p in enumerate(pids)]); counts = np.array([len(cal[p]) for p in pids], float)
    split = np.array([half(p) for p in pids]); out = {'n_papers': len(pids), 'n_sentences': int(len(scores)), 'thresholds': []}
    for f in TARGETS:
        t = threshold_at(scores, f)
        reps = [weighted_threshold(scores, owner, np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids)).astype(float), f) for _ in range(B // 4)]
        a = np.concatenate([cal[p] for p, s in zip(pids, split) if s == 0]); ta = threshold_at(a, f)
        fb = np.array([(cal[p] > ta).sum() for p, s in zip(pids, split) if s == 1], float); cb = np.array([len(cal[p]) for p, s in zip(pids, split) if s == 1], float)
        flags = np.array([(cal[p] > t).sum() for p in pids], float)
        out['thresholds'].append({'target_fpr': f, 'threshold': t, 'threshold_ci95': [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))],
                                  'heldout_threshold_half_a': ta, 'heldout_fpr': float(fb.sum() / cb.sum()), 'heldout_ci95_paper_bootstrap': paper_boot_rate(fb, cb),
                                  'heldout_papers': int(len(fb)), 'heldout_sentences': int(cb.sum()), 'papers_with_any_flag': float(np.mean(flags > 0)),
                                  'per_paper_flag_share_p50': float(np.median(flags / np.maximum(counts, 1))),
                                  'per_paper_flag_share_p99': float(np.quantile(flags / np.maximum(counts, 1), .99))})
    t1 = next(x['threshold'] for x in out['thresholds'] if x['target_fpr'] == 0.01)
    return out, {p: float((cal[p] > t1).mean()) if len(cal[p]) else 0.0 for p in pids}, t1


def load_tokens(out_dir):
    tok = {}
    for f in sorted((Path(out_dir) / 'tokens').glob('tokens-*.npz')):
        z = np.load(f)
        for k in z.files:
            pid, kind = k.rsplit('/', 1); tok.setdefault(pid, {})[kind] = z[k]
    docs = {}
    for f in sorted((Path(out_dir) / 'data').glob('summary-*.parquet')):
        for r in pq.read_table(f).to_pylist(): docs[r['id']] = r
    return tok, docs


def main(out_cal, cal_text, old_text, humancal, out_json):
    tok, docs = load_tokens(out_cal)
    texts = {}
    for d in (cal_text, old_text):
        for f in sorted((Path(d) / 'data').glob('*.parquet')):
            for r in pq.read_table(f, columns=['id', 'text']).to_pylist(): texts[r['id']] = r['text']
    hc = Path(humancal)
    overlap = json.loads((hc / 'overlap.json').read_text())['papers']
    meta = {p['id']: p for p in json.loads((hc / 'selection.json').read_text())['papers']}
    keep = {p for p, v in overlap.items() if v['shared_unique'] <= 0 and not v['key_hit']}
    old_ids = {r['id'] for f in sorted((Path(old_text) / 'data').glob('*.parquet')) for r in pq.read_table(f, columns=['id']).to_pylist()}
    sent = {pid: sentence_scores(texts[pid], d['offsets'], d['probs']) for pid, d in tok.items()}
    missing = sorted((set(overlap) | old_ids) - set(sent))
    res = {'scored_papers': len(sent), 'missing': len(missing), 'kept_new': len(keep & set(sent)), 'old': len(old_ids & set(sent)), 'variants': {}}
    for name, ids in [('kept+old (as Atlas calibration)', [p for p in sent if p in keep] + [p for p in sent if p in old_ids]),
                      ('all new 2,299 + old', [p for p in sent if p in overlap or p in old_ids])]:
        cal = {('old:' + p if p in old_ids else p): sent[p] for p in ids}
        block, share, t1 = sentence_block(cal)
        groups = defaultdict(list)
        for p, s in share.items():
            if p.startswith('old:'): groups[('archive', 2022)].append(s)
            else: groups[(meta[p]['venue'], meta[p]['year'])].append(s)
        dv = {c: np.array([docs[p.replace('old:', '')][c] for p in cal if docs.get(p.replace('old:', ''), {}).get(c) is not None], float) for c in ('document_prob', 'token_prob_mean')}
        block['document_threshold_1pct'] = {c: threshold_at(v, 0.01) for c, v in dv.items()}
        block['groups'] = [{'venue': v, 'year': y, 'papers': len(s), 'flagged_share_mean': float(np.mean(s)), 'flagged_share_p50': float(np.median(s)),
                            'flagged_share_p25': float(np.quantile(s, .25)), 'flagged_share_p75': float(np.quantile(s, .75))} for (v, y), s in sorted(groups.items())]
        block['human_flag_share_hist'] = np.histogram(list(share.values()), bins=20, range=(0, 0.2))[0].tolist()
        res['variants'][name] = block
        print(name, json.dumps({k: block[k] for k in ('n_papers', 'n_sentences')}), json.dumps(block['thresholds'][1]), flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
