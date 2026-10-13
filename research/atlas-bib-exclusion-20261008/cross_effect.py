"""Cross-effect of counting modes on the human false-positive rate.

For each calibration mode m, set the 1%-FPR sentence threshold on half A of the 2,403 human papers (cal-split-v1, as
cal_analyze), then measure the flag rate on half B under every counting mode n. The diagonal is the honest held-out
FPR of each mode's own threshold; off-diagonal cells show what happens when a threshold is applied to sentences it was
not calibrated on. Intervals resample half-B papers (flags cluster within papers). Also reports, per cell, the median
and 99th-percentile per-paper flagged share on half B.
Usage: cross_effect.py WORK OUT.json
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import ca, human_scores_by_mode  # noqa: E402
from bib import MODES  # noqa: E402


def main(work, out_json):
    by_mode, _, _ = human_scores_by_mode(work)
    split = {p: ca.half(p) for p in by_mode['all']}
    res = {'split': 'cal-split-v1 (threshold on half A, rate on half B)', 'thresholds_half_a': {}, 'cells': []}
    for m in MODES:
        a = np.concatenate([s for p, s in by_mode[m].items() if split[p] == 0]); t = ca.threshold_at(a, 0.01)
        res['thresholds_half_a'][m] = t
        for n in MODES:
            b = [s for p, s in by_mode[n].items() if split[p] == 1]
            flags = np.array([(s > t).sum() for s in b], float); counts = np.array([len(s) for s in b], float)
            share = flags / np.maximum(counts, 1)
            res['cells'].append({'threshold_mode': m, 'applied_to': n, 'threshold': t, 'fpr': float(flags.sum() / counts.sum()),
                                 'fpr_ci95': ca.paper_boot_rate(flags, counts), 'papers': len(b), 'sentences': int(counts.sum()),
                                 'paper_share_p50': float(np.median(share)), 'paper_share_p99': float(np.quantile(share, .99))})
    Path(out_json).write_text(json.dumps(res, indent=1))
    print('FPR on half B (rows: threshold calibrated with mode; columns: sentences counted)')
    print('            ' + ''.join(f'{n:>10}' for n in MODES))
    for m in MODES:
        print(f'{m:>6} {res["thresholds_half_a"][m]:.5f}' + ''.join(f'{c["fpr"]:>9.2%} ' for c in res['cells'] if c['threshold_mode'] == m))


if __name__ == '__main__':
    main(*sys.argv[1:3])
