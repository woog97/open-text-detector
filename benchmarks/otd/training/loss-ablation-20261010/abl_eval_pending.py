"""Print how many eval_final.py jobs of a run are still unclaimed (0 when there is nothing for another worker to take).
Mirrors eval_final.main for a single-pass run: the sweep job on the selected checkpoint plus one job per eval set whose input
exists. A job counts as taken when its result .json or its .claim file exists. Usage: abl_eval_pending.py TAG"""
import json, sys
from pathlib import Path
d = Path('sweeps') / sys.argv[1]; ev = d / 'eval'; I = Path('eval-inputs')
ck = json.loads((d / 'stage2-selection.json').read_text())['checkpoint'].split('-adapters')[0]
sets = {'heldout-writers': 'heldout-score-rows.jsonl.gz', 't21-heldout-v2': 't21-heldout-score-rows-v2.jsonl.gz', 'fable-polish': 'polish-eval-v1.jsonl.gz',
        'fable-humanizer': 'humanizer-eval-v1.jsonl.gz', 'fable-cowrite': 'cowrite-eval-v1.jsonl.gz', 'cross-model-s500': 'hetero', 'wide1-heldout': 'wide1-heldout-rows.jsonl.gz'}
jobs = [ck] + [f'{n}-{ck}' for n, p in sets.items() if (I / p).exists()]
print(sum(not (ev / f'{j}.json').exists() and not (ev / f'{j}.claim').exists() for j in jobs))
