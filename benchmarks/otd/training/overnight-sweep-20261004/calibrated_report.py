"""Held-out operating points: fit sentence cutoffs on calibration windows, apply them to the sweep test half.

Inputs (fetched from the hosts into results/calibration/<tag>/): <ckpt>-calibration.npz and <ckpt>-sentences.npz.
For target FPRs 0.5/1/2%, the cutoff is the calibration human-sentence quantile; we report the realized human
FPR on test-half paper sentences and recall on edits (all/small/one/para) and paper_v3 sentences.
"""
import gzip, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = [json.loads(l) for l in gzip.open(HERE / 'sweep-eval-rows.jsonl.gz', 'rt')]
SIZES = {'all': ('sentence', 'two_sentence', 'paragraph_concise', 'paragraph_v3'), 'small': ('sentence', 'two_sentence'),
         'one': ('sentence',), 'para': ('paragraph_concise', 'paragraph_v3')}


def cutoff(human_scores, fpr):
    s = np.sort(human_scores)[::-1]; k = int(np.floor(fpr * len(s)))
    return float(s[k]) if k < len(s) else -np.inf


def analyze(cal_path, sent_path, split='test'):
    c = np.load(cal_path); z = np.load(sent_path)
    hum_cal = c['scores'][c['labels'] == 0]
    neg, pos, v3p, v3n = [], {k: [] for k in SIZES}, [], []
    for i, r in enumerate(ROWS):
        if r['split'] != split:
            continue
        sl = z[str(i)]
        if r['slot'] in ('edit_context', 'human_untouched', 'human_paper'):
            neg += [s for s, l in sl if l == 0]
        if r['slot'] == 'edit_context':
            for k, conds in SIZES.items():
                if r['condition'] in conds:
                    pos[k] += [s for s, l in sl if l == 1]
        if r['slot'] == 'v3':
            v3p += [s for s, l in sl if l == 1]; v3n += [s for s, l in sl if l == 0]
    neg, v3p, v3n = map(np.asarray, (neg, v3p, v3n))
    out = {}
    for f in (0.005, 0.01, 0.02):
        t = cutoff(hum_cal, f)
        out[f'{f:.3f}'] = {'cutoff': t, 'test_human_fpr': float(np.mean(neg > t)),
                           **{f'recall_{k}': float(np.mean(np.asarray(v) > t)) for k, v in pos.items()},
                           'v3_recall': float(np.mean(v3p > t)), 'v3_human_fpr': float(np.mean(v3n > t))}
    return out


if __name__ == '__main__':
    base = HERE / 'results' / 'calibration'; report = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        for cal in sorted(d.glob('*-calibration.npz')):
            ck = cal.name.replace('-calibration.npz', ''); sent = d / f'{ck}-sentences.npz'
            if sent.exists():
                report[f'{d.name}/{ck}'] = analyze(cal, sent)
    (HERE / 'results' / 'calibrated-operating-points.json').write_text(json.dumps(report, indent=1))
    print(f"{'run':32s} {'cut':>6s} {'testFPR':>8s} {'all':>6s} {'small':>6s} {'one':>6s} {'para':>6s} {'v3':>6s} {'v3FPR':>6s}   (target 1% FPR)")
    for k, v in report.items():
        m = v['0.010']
        print(f"{k:32s} {m['cutoff']:6.3f} {m['test_human_fpr']:8.4f} {m['recall_all']:6.3f} {m['recall_small']:6.3f} {m['recall_one']:6.3f} {m['recall_para']:6.3f} {m['v3_recall']:6.3f} {m['v3_human_fpr']:6.3f}")
