"""Set up MoE scoring for the mix-grid runs in /tmp/mixgrid-20261007 and start one worker per free GPU (4-7 now; 0-1
after moe-mixA finishes). GPUs 2-3 are never used. The sweep_eval copy imports modeling_sweep (the module the runs were
trained with) and uses micro-batch 16."""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent
FILES = {n: (H / n).read_text() for n in ['sweep_one.py', 'score_worker_mg.py']}
code = r'''
import json, shutil, subprocess, sys
from pathlib import Path
R = Path('/tmp/mixgrid-20261007'); S = R / 'sweeps'; F = Path('/tmp/final-20261007'); T2 = Path('/tmp/pangram-t2-20261006')
src = (F / 'sweep_eval.py').read_text()
assert 'from modeling import Detector, collate' in src and 'model.physical_microbatch = 8' in src and 'def predict(rows, model, tok, batch_size=8)' in src
src = src.replace('from modeling import Detector, collate', 'from modeling_sweep import Detector, collate').replace('model.physical_microbatch = 8', 'model.physical_microbatch = 16')
src = src.replace('def predict(rows, model, tok, batch_size=8)', 'def predict(rows, model, tok, batch_size=16)')
(R / 'sweep_eval.py').write_text(src); shutil.copy(F / 'cross_model_eval.py', R / 'cross_model_eval.py')
for n, s in FILES.items(): (R / n).write_text(s)
if not (S / 'sweep-eval-rows.jsonl.gz').exists(): shutil.copy(T2 / 'sweeps/sweep-eval-rows.jsonl.gz', S / 'sweep-eval-rows.jsonl.gz')
for p in ['inputs/t21-heldout-score-rows-v2-clean-for-t21.jsonl.gz', 'inputs/heldout-score-rows.jsonl.gz', 'inputs/t21-heldout-score-rows.jsonl.gz', 'hetero',
          'inputs/polish-eval-v1.jsonl.gz', 'inputs/humanizer-eval-v1.jsonl.gz', 'inputs/cowrite-eval-v1.jsonl.gz']:
    assert (T2 / p).exists(), p
assert not list(S.glob('*/mg-*.claim')), 'already started'
pids = {}
for g in (4, 5, 6, 7):
    pids[g] = subprocess.Popen([sys.executable, '-u', str(R / 'score_worker_mg.py'), str(g)], cwd=R, stdout=open(S / f'score-mg-gpu{g}.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True).pid
for g in (0, 1):
    pids[g] = subprocess.Popen([sys.executable, '-u', str(R / 'score_worker_mg.py'), str(g), 'moe-mixA'], cwd=R, stdout=open(S / f'score-mg-gpu{g}.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True).pid
(S / 'score-mg-pids.json').write_text(json.dumps(pids)); print('started', pids)
'''
run('FILES=' + repr(FILES) + '\n' + code, timeout=180)
