"""Extend the MoE Calibration page's sentence ROC curves down to FPR 1e-6 (Oct 11 2026).

Recomputes only sentence_curves[*].fpr/tpr in app/lib/model-calibration/qwen36-35b-a3b-final*.json on the grid
unique(geomspace(1e-6, 1, 121) ∪ linspace(0, 1, 101)) (the old grid started at 1e-4 with the same 20 points per decade).
Positives and negatives are those of the stored curves: suite edit sentences scored in context
(build_moe_calibration.sets_for on suite-sentences-moe.json.gz) against the human calibration sentences each counting
mode keeps (recompute.human_scores_by_mode: all 2,403 papers, i.e. the curves' n_human predates the Oct 11 removal of the
three archive papers; the thresholds do not). Same roc() as build_moe_calibration.py.

Before writing, every curve must reproduce the stored one: tpr at each old grid point equal to 1e-5, and n_ai, n_human,
auc (5 decimals) equal. Otherwise nothing is written.
Usage: extend_roc_grid.py WORK SUITE_SENTENCES.json.gz JSON_DIR [--write]
WORK as for recompute.py (out-cal/, cal-text/, old-text/, overlap.json, selection.json).
"""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008')); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import build_moe_calibration as bm  # noqa: E402
import recompute as rc  # noqa: E402

OLD_GRID = bm.FPR_GRID
NEW_GRID = np.unique(np.r_[np.geomspace(1e-6, 1, 121), np.linspace(0, 1, 101)])
FILES = {'all': 'qwen36-35b-a3b-final.json', 'nobib': 'qwen36-35b-a3b-final-nobib.json',
         'noapp': 'qwen36-35b-a3b-final-noapp.json', 'body': 'qwen36-35b-a3b-final-body.json'}
SETS = {'One-sentence edits': 'papers_one', 'Two-sentence edits': 'papers_two', 'Paragraph edits': 'papers_para'}


def fpr_value(v):
    """Old points keep round(v, 6) as stored; points below 1e-4 keep 3 significant figures."""
    return round(float(v), 6) if v >= 1e-4 or v == 0 else float(f'{v:.3g}')


def tpr_on(pos, neg, grid):
    bm.FPR_GRID = grid
    try: return bm.roc(pos, neg, 0.0)
    finally: bm.FPR_GRID = OLD_GRID


def main(work, suite_path, json_dir, write=False):
    suite = json.load(gzip.open(suite_path, 'rt')); pos = defaultdict(list)
    for r in suite:
        if 'sentences' not in r: continue
        for k in bm.sets_for(r): pos[k] += [s for s, g in r['sentences'] if g == 1]
    by_mode, _, _ = rc.human_scores_by_mode(work)
    ok = True; updated = {}
    for mode, fname in FILES.items():
        path = Path(json_dir) / fname; data = json.loads(path.read_text())
        neg = np.concatenate(list(by_mode[mode].values()))
        for c in data['sentence_curves']:
            p = pos[SETS[c['name']]]
            old = tpr_on(p, neg, OLD_GRID); new = tpr_on(p, neg, NEW_GRID)
            new_fpr = [fpr_value(v) for v in NEW_GRID]
            # new-grid points at fpr >= 1e-4 must be the old points with the old values
            hi = [(f, t) for f, t, g in zip(new_fpr, new['tpr'], NEW_GRID) if g >= 1e-4 * (1 - 1e-9) or g == 0]
            checks = {'n_ai': old['n_ai'] == c['n_ai'], 'n_human': old['n_human'] == c['n_human'], 'auc': bm.r5(old['auc']) == c['auc'],
                      'old_fpr': old['fpr'] == c['fpr'],
                      'old_tpr': len(old['tpr']) == len(c['tpr']) and max(abs(a - b) for a, b in zip(old['tpr'], c['tpr'])) <= 1e-5,
                      'new_at_old_points': [f for f, _ in hi] == c['fpr'] and max(abs(t - b) for (_, t), b in zip(hi, c['tpr'])) <= 1e-5}
            print(f"{mode:6s} {c['name']:20s} n_ai {old['n_ai']} n_human {old['n_human']:,} auc {bm.r5(old['auc'])} "
                  f"{'OK' if all(checks.values()) else 'FAIL ' + str([k for k, v in checks.items() if not v])}  "
                  f"tpr@1e-5 {new['tpr'][new_fpr.index(1e-5)]} tpr@1e-6 {new['tpr'][new_fpr.index(1e-6)]}", flush=True)
            ok &= all(checks.values())
            c['fpr'], c['tpr'] = new_fpr, new['tpr']
        updated[path] = data
    if not ok: sys.exit('not reproduced; nothing written')
    if write:
        for path, data in updated.items(): path.write_text(json.dumps(data, separators=(',', ':')))
        print('wrote', len(updated), 'files')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], '--write' in sys.argv[4:])
