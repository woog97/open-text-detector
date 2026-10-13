"""Space-side: recompute the Calibration page's by-year rows (Atlas fast10) at new thresholds.

Usage: year_groups_space.py SENT_T HUMAN_P99 DOC_T OUT.json
  SENT_T     sentence threshold at 1% FPR; HUMAN_P99 the 99th percentile of the per-paper flagged share among the
             calibration papers at SENT_T; DOC_T the document-head threshold at 1% FPR.
Inputs are the saved Atlas scores (no inference): year-diagnostic-v1 (2023-2026 papers, training overlap already
removed) and the ICLR 2027 classification. Output has the group-summary.json schema of the Oct 6 year diagnostic
(the HUMAN row is added by export_calibration.py from the new calibration analysis).
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

C = Path('/data/workspace/classifications')
SETS = [('year', C / 'calibration-inputs/year-diagnostic-v1/data', C / 'year-diagnostic-v1-20261006/qwen35-4b-fast10'),
        ('iclr2027', None, C / 'iclr2027-clean-v2-20261005/qwen35-4b-fast10')]


def main(t, h99, doc_t, out):
    t, h99, doc_t = float(t), float(h99), float(doc_t)
    groups = defaultdict(lambda: {'flag': [], 'doc': [], 'tok': []})
    for kind, meta_dir, run in SETS:
        if meta_dir:
            meta = {r['id']: f"{r['source']}|{r['conference']}|{r['year']}"
                    for f in sorted(meta_dir.glob('train-*.parquet')) for r in pq.read_table(f, columns=['id', 'source', 'conference', 'year']).to_pylist()}
        docs = {r['id']: r for f in sorted((run / 'data').glob('summary-*.parquet'))
                for r in pq.read_table(f, columns=['id', 'document_prob', 'token_prob_mean']).to_pylist()}
        for f in sorted((run / 'sentences').glob('sentences-*.npz')):
            z = np.load(f)
            for k in z.files:
                pid, part = k.rsplit('/', 1)
                if part != 'scores': continue
                g = meta[pid] if meta_dir else 'ICLR 2027 (all)'
                s = z[k].astype(float)
                if not len(s): continue
                groups[g]['flag'].append(float((s > t).mean()))
                groups[g]['doc'].append(docs[pid]['document_prob']); groups[g]['tok'].append(docs[pid]['token_prob_mean'])
    rows = []
    for g, v in sorted(groups.items()):
        f, d, k = (np.asarray(v[x], float) for x in ('flag', 'doc', 'tok'))
        rows.append({'group': g, 'n': int(len(f)), 'flag_p50': float(np.median(f)), 'flag_p25': float(np.quantile(f, .25)),
                     'flag_p75': float(np.quantile(f, .75)), 'over_h99': float(np.mean(f > h99)), 'doc_p50': float(np.median(d)),
                     'doc_over': float(np.mean(d > doc_t)), 'tok_p50': float(np.median(k))})
    Path(out).write_text(json.dumps(rows, indent=1))
    print(json.dumps([{k: r[k] for k in ('group', 'n', 'flag_p50', 'over_h99')} for r in rows]))


if __name__ == '__main__':
    main(*sys.argv[1:5])
