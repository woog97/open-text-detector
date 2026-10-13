"""Score <-> human false-positive-rate map and extra operating points for the Oct 10 Calibration page and reader sliders.

For each counting mode the site offers (nobib, body), on the same 2,400 human calibration papers as the published cutoffs
(recompute.human_scores_by_mode minus recalibrate.EXCLUDE):
- score_map: for 140 log-spaced false-positive rates from one human sentence (1/n) to 0.5, plus n and the highest human score, the sentence score cal_analyze.threshold_at gives
  (so the 0.1% and 1% points equal the published cutoffs);
- operating points at 0.01% and 5% with the same held-out check as the published ones (cal_analyze.sentence_block:
  cutoff set on half A of the papers, rate measured on half B, paper bootstrap). 0.1% and 1% are taken unchanged from
  recal-final-moe.json, so the published entries do not move.
Writes the four app/lib/model-calibration/qwen36-35b-a3b-oct10-{nobib,body}.json fields `score_map` and extra
`thresholds` rows, and app/lib/score-fpr.ts (the reader's lookup).
Usage: score_map_oct10.py WORK_NEW RECAL.json APP_DIR
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; R = HERE.parent; MAIN = R  # bib-exclusion code is tracked since Oct 12
sys.path[:0] = [str(R / 'atlas-calibration-fix-20261011'), str(MAIN / 'atlas-bib-exclusion-20261008'), str(R / 'atlas-rescore-moe-20261007')]
import cal_analyze as ca  # noqa: E402
import recompute as rc  # noqa: E402
from recalibrate import EXCLUDE  # noqa: E402

READOUT_TS = '/** Short label for a human false-positive rate of a sentence cutoff, e.g. "≈0.12% of human sentences". */\nexport function fprReadout(mode:string,score:number){\n if(mode!==\'nobib\'&&mode!==\'body\')return \'\';\n const m=SCORE_FPR[mode],f=fprForScore(mode,score);\n if(score>m.max)return \'above every human sentence\';\n if(score>=m.score[0])return `≈${(m.fpr[0]*100).toFixed(5)}% (one human sentence)`;\n if(score<=m.score[m.score.length-1])return `>${(m.fpr[m.fpr.length-1]*100).toFixed(0)}% of human sentences`;\n const v=f>=0.01?(f*100).toFixed(1):f>=0.001?(f*100).toFixed(2):f>=0.0001?(f*100).toFixed(3):(f*100).toFixed(4);\n return `≈${v}% of human sentences`;\n}\n'
GRID_TOP = 0.5  # the map runs from one human sentence (1/n) up to 50%


def main(work, recal_json, app):
    app = Path(app); recal = json.loads(Path(recal_json).read_text())['modes']
    by_mode, _, _ = rc.human_scores_by_mode(work); ts_lines = []
    for mode in ('nobib', 'body'):
        cal = {k: v for k, v in by_mode[mode].items() if k not in EXCLUDE}; s = np.concatenate(list(cal.values()))
        pub = {t['target_fpr']: t for t in recal[mode]['thresholds']}
        for f in (0.001, 0.01): assert ca.threshold_at(s, f) == pub[f]['threshold'], (mode, f)
        # sentence_block also needs the 1% target (it returns the 1% cutoff); only its 0.01% row is used.
        ca.TARGETS = [0.0001, 0.01]; ca.rng = np.random.default_rng(20261012); block, _, _ = ca.sentence_block(cal)
        extra = [next(t for t in block['thresholds'] if t['target_fpr'] == 0.0001), pub[0.05]]
        grid = [float(x) for x in np.logspace(np.log10(1 / len(s)), np.log10(GRID_TOP), 140)]
        smap = {'fpr': grid, 'score': [float(ca.threshold_at(s, f)) for f in grid], 'n_sentences': int(len(s)), 'max_score': float(s.max())}
        p = app / 'lib/model-calibration' / f'qwen36-35b-a3b-oct10-{mode}.json'; d = json.loads(p.read_text())
        rows = {t['target_fpr']: t for t in d['thresholds']}
        for t in extra:
            rows[t['target_fpr']] = {'target_fpr': t['target_fpr'], 'threshold': t['threshold'], 'threshold_ci95': t['threshold_ci95'],
                                     'heldout_fpr': t['heldout_fpr'], 'heldout_ci95': t['heldout_ci95_paper_bootstrap'],
                                     'heldout_sentences': t['heldout_sentences'], 'heldout_papers': t['heldout_papers'],
                                     'stable': rows[0.001]['stable']}
        d['thresholds'] = [rows[f] for f in sorted(rows)]; d['score_map'] = smap
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + '\n')
        print(mode, len(s), [(t['target_fpr'], round(t['threshold'], 6), round(t['heldout_fpr'], 6)) for t in d['thresholds']])
        ts_lines.append(f" {mode}:{{fpr:{json.dumps([round(x, 10) for x in grid])},score:{json.dumps([round(x, 8) for x in smap['score']])},max:{smap['max_score']!r}}},")
    (app / 'lib/score-fpr.ts').write_text(
        '/* Oct 10 model: sentence score <-> share of human sentences at or above it, on the 2,400 human calibration papers, per\n'
        '   counting mode (research/atlas-oct10-20261012/score_map_oct10.py). Generated; do not edit by hand. */\n'
        'export const SCORE_FPR:Record<\'nobib\'|\'body\',{fpr:number[];score:number[];max:number}>={\n' + '\n'.join(ts_lines) + '\n};\n'
        '/** Human false-positive rate of a sentence cutoff, interpolated on a log scale; clamped to the map\'s range. */\n'
        'export function fprForScore(mode:\'nobib\'|\'body\',score:number){const m=SCORE_FPR[mode];if(score>=m.score[0])return m.fpr[0];const n=m.score.length;if(score<=m.score[n-1])return m.fpr[n-1];\n'
        ' let i=0;while(i+1<n&&m.score[i+1]>score)i++;const a=m.score[i],b=m.score[i+1],t=(Math.log(score)-Math.log(a))/(Math.log(b)-Math.log(a));\n'
        ' return Math.exp(Math.log(m.fpr[i])+t*(Math.log(m.fpr[i+1])-Math.log(m.fpr[i])));}\n'
        + READOUT_TS)


if __name__ == '__main__':
    main(*sys.argv[1:4])
