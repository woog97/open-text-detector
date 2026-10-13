"""Paper-level flagged shares under four counting modes (bibliography and appendices in or out), for the final MoE on the
Paper Atlas (no model rerun). Modes: bib.MODES.

1. calibrate: the 1%-FPR sentence threshold on the same 2,403 human papers as cal_analyze.py, from the stored token
   scores (out-cal) and texts, once per mode, keeping only the sentences the mode counts ('all' must reproduce the
   published 0.0058389). Same thresholding, interval and held-out code (cal_analyze).
2. atlas: for each MoE detail object staged for the Atlas (staging/objects-*.tar; the published combined objects hold
   the same MoE report next to the 4B one), count the MoE report's sentences outside
   the sentences each mode counts and those above that mode's threshold (sentences without letters or digits stay
   excluded, as before). Also recounts the published summary with the old rule as a check.
Usage: recompute.py calibrate WORK OUT.json
       recompute.py atlas WORK CATALOGUE.json CALIBRATION.json OUT.json TAR [TAR ...]   (WORK also holds atlas-details-index-moe.parquet)
WORK holds out-cal/, cal-text/data/*.parquet (human-calibration clean-data.tar), old-text/data/*.parquet,
overlap.json and selection.json.
"""
import gzip, json, sys, tarfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import cal_analyze as ca  # noqa: E402
from bib import MODES, included, section_of, sections  # noqa: E402

PUBLISHED_T = 0.005838871002197266


def sentence_scores_spans(text, off, prob):
    """cal_analyze.sentence_scores, also returning each sentence's start."""
    off = np.asarray(off).reshape(-1, 2); prob = np.asarray(prob, dtype=np.float64)
    valid = np.array([bool(text[a:b].strip()) for a, b in off], dtype=np.float64)
    cs = np.r_[0, np.cumsum(prob * valid)]; cv = np.r_[0, np.cumsum(valid)]; out = []; starts = []
    for m in ca.SENT.finditer(text):
        if not m.group().strip(): continue
        a, b = m.start(), m.end(); lo = np.searchsorted(off[:, 1], a, side='right'); hi = np.searchsorted(off[:, 0], b, side='left'); n = cv[hi] - cv[lo]
        if hi > lo and n > 0: out.append((cs[hi] - cs[lo]) / n); starts.append(a)
    return np.array(out), starts


def human_scores_by_mode(work):
    """{mode: {paper key: sentence scores counted under that mode}} for the 2,403 human calibration papers, plus
    (bibliography-found count, per-paper section shares)."""
    work = Path(work); tok, _ = ca.load_tokens(work / 'out-cal'); texts = {}
    for d in ('cal-text', 'old-text'):
        for f in sorted((work / d / 'data').glob('*.parquet')):
            for r in pq.read_table(f, columns=['id', 'text']).to_pylist(): texts[r['id']] = r['text']
    overlap = json.loads((work / 'overlap.json').read_text())['papers']
    keep = {p for p, v in overlap.items() if v['shared_unique'] <= 0 and not v['key_hit']}
    old_ids = {r['id'] for f in sorted((work / 'old-text' / 'data').glob('*.parquet')) for r in pq.read_table(f, columns=['id']).to_pylist()}
    by_mode = {m: {} for m in MODES}; found = 0; shares = {'bib': [], 'app': []}
    for pid, d in tok.items():
        if not (pid in keep or pid in old_ids): continue
        s, starts = sentence_scores_spans(texts[pid], d['offsets'], d['probs']); spans, tail = sections(texts[pid])
        sec = [section_of(a, spans, tail) for a in starts]
        key = 'old:' + pid if pid in old_ids else pid; found += bool(spans)
        for m in MODES: by_mode[m][key] = s[np.array([included(x, m) for x in sec], bool)] if len(sec) else s
        for k in shares: shares[k].append(sec.count(k) / max(len(sec), 1))
    return by_mode, found, shares


def calibrate(work, out_json):
    by_mode, found, shares = human_scores_by_mode(work)
    res = {'papers': len(by_mode['all']), 'bibliography_found': found, 'modes': {m: {'include_bibliography': MODES[m][0], 'include_appendices': MODES[m][1]} for m in MODES},
           'section_share_median': {k: float(np.median(v)) for k, v in shares.items()}}
    for m, cal in by_mode.items():
        block, share, t1 = ca.sentence_block(cal)
        res['modes'][m].update(block, human_paper_flag_share_p50=float(np.median(list(share.values()))),
                               human_paper_flag_share_p99=float(np.quantile(list(share.values()), .99)))
        print(m, block['n_papers'], block['n_sentences'], json.dumps(block['thresholds'][1]), flush=True)
    assert abs(res['modes']['all']['thresholds'][1]['threshold'] - PUBLISHED_T) < 1e-9, 'did not reproduce the published threshold'
    Path(out_json).write_text(json.dumps(res, indent=1))


def atlas(work, catalogue, calibration, out_json, tars):
    cal = json.loads(Path(calibration).read_text())['modes']
    thresholds = {m: next(t['threshold'] for t in cal[m]['thresholds'] if t['target_fpr'] == 0.01) for m in MODES}
    by_pdf = {p['pdf_key'][7:-4]: p for p in json.loads(Path(catalogue).read_text())['items']}
    staged = {r['detail_key']: r['pdf_sha256'] for r in pq.read_table(Path(work) / 'atlas-details-index-moe.parquet', columns=['detail_key', 'pdf_sha256']).to_pylist()}
    by_key = {k: by_pdf[h] for k, h in staged.items() if h in by_pdf}
    out, mismatch, missing = {}, 0, 0
    for tar in tars:
        with tarfile.open(tar) as tf:
            for m in tf:
                if not m.isfile(): continue
                paper = by_key.get('atlas-public/' + m.name)
                if paper is None: missing += 1; continue
                d = json.loads(gzip.decompress(tf.extractfile(m).read()))
                r = next(x for x in d['reports'] if x['id'].startswith('qwen36-35b-a3b-')); t = r['text']; spans, tail = sections(t)
                counts = {mode: [0, 0] for mode in MODES}; size = {'body': 0, 'bib': 0, 'app': 0}; old_total = old_flagged = 0
                for s in r['result']['segments']:
                    if not any(c.isalnum() for c in t[s['start']:s['end']]): continue
                    old_total += 1; old_flagged += s['label'] == 'ai_evidence'
                    sec = section_of(s['start'], spans, tail); size[sec] += 1
                    for mode in MODES:
                        if included(sec, mode): counts[mode][1] += 1; counts[mode][0] += s['score'] > thresholds[mode]
                pub = (paper.get('score_summaries') or {}).get('qwen36-35b-a3b') or {}
                mismatch += (pub.get('flagged'), pub.get('total')) != (old_flagged, old_total)
                out[paper['id']] = {'modes': counts, 'sections': [size['body'], size['bib'], size['app']]}
        print(f'{Path(tar).name}: {len(out):,} papers so far', flush=True)
    Path(out_json).write_text(json.dumps({'thresholds': thresholds, 'papers': out, 'published_summary_mismatches': mismatch, 'objects_not_in_catalogue': missing}))
    print(f'done: {len(out):,} papers; published-summary mismatches {mismatch}; objects not in catalogue {missing}', flush=True)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'calibrate': calibrate(*args)
    else: atlas(args[0], args[1], args[2], args[3], args[4:])
