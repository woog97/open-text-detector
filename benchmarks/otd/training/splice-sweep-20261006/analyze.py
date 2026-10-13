"""Compare splice arms (SPH, SPG) with baseline arm A (q4b-A-s1..3) on the sweep eval set, final stage-2 epoch.
Metrics: test-half recall at 1% FPR (test-half cutoff, as in the sweep) and AUROC; plus the held-out method
(cutoff at 1% FPR on DEV human sentences, applied to TEST), with realized test human FPR."""
import gzip, json, math, re, statistics, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
SW = ROOT / 'benchmarks/otd/training/overnight-sweep-20261004'
HON = (ROOT / 'research/evaluation/backbone-results-20261003/compare_old_new.py').read_text()
ns = {}
src = (SW / 'sweep_eval.py').read_text()
for name in ['sentences', 'token_labels', 'sentence_rows', 'auroc', 'recall_at', 'metrics']:
    exec(re.search(rf'\ndef {name}\(.*?(?=\n\n\n|\n\n@|\nif __name__)', src, re.S).group(0), {'np': np, 're': re, 'math': math, **ns}, ns)
g = {'np': np, **ns}
exec(re.search(r"\nHUMAN_SLOTS = .*?\n\n\ndef honest\(.*?(?=\n\n\n)", HON, re.S).group(0), g)
rows = [json.loads(l) for l in gzip.open(SW / 'sweep-eval-rows.jsonl.gz', 'rt')]
NEW = Path(sys.argv[1])


def load(p):
    z = np.load(p); sents = [[(float(s), int(l)) for s, l in z[str(i)]] for i in range(len(rows))]; return sents, [float(x) for x in z['doc']]


def score(sents, docs):
    scored = [{'slot': r['slot'], 'split': r['split'], 'label': r['label'], 'condition': r.get('condition'), 'doc': d, 'sents': s} for r, s, d in zip(rows, sents, docs)]
    t = ns['metrics'](scored, 'test'); h = g['honest'](rows, sents, docs)
    out = {f'{k}_r1': t[f'edits_{k}']['recall_at_1pct'] for k in ['one', 'two', 'small', 'para', 'all']}
    out.update({f'{k}_auc': t[f'edits_{k}']['auroc'] for k in ['one', 'two', 'small', 'para', 'all']})
    out.update({'hold_small': h['recall_small'], 'hold_one': h['recall_one'], 'hold_all': h['recall_all'], 'hold_para': h['recall_para'],
                'hold_test_fpr': h['test_human_fpr'], 'fpr_at_05': t['human_sentence_fpr_at_05'], 'v3_r1': t['v3_sentences']['recall_at_1pct'],
                'standalone_r1': t['standalone_rewrites']['recall_at_1pct'], 'public_auc': t['public_docs']['auroc']})
    return out


arms = {'A (baseline)': [SW / f'results/calibration/q4b-A-s{s}/stage2-epoch2-sentences.npz' for s in (1, 2, 3)],
        'SPH (half papers -> splices)': [NEW / f'q4b-SPH-s{s}/eval/stage2-epoch2-sentences.npz' for s in (1, 2, 3)],
        'SPG (gradtex -> splices)': [NEW / f'q4b-SPG-s{s}/eval/stage2-epoch2-sentences.npz' for s in (1, 2, 3)]}
res = {a: [score(*load(p)) for p in ps if p.exists()] for a, ps in arms.items()}
json.dump(res, open(NEW / 'splice-results.json', 'w'), indent=1)
keys = ['one_r1', 'two_r1', 'small_r1', 'para_r1', 'all_r1', 'one_auc', 'small_auc', 'all_auc', 'hold_one', 'hold_small', 'hold_all', 'hold_test_fpr', 'fpr_at_05', 'v3_r1', 'standalone_r1', 'public_auc']
print('metric'.ljust(15) + ''.join(a.split(' ')[0].rjust(18) for a in res))
for k in keys:
    line = k.ljust(15)
    for a, rs in res.items():
        v = [r[k] for r in rs if r[k] is not None]
        line += (f'{statistics.mean(v):.3f}±{statistics.pstdev(v):.3f}' if v else '-').rjust(18)
    print(line)
print({a: len(rs) for a, rs in res.items()})
