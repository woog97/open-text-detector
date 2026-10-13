"""Push the soft-label arm to the Space and start run_soft_t21.sh (it waits for T2.1 scoring to leave GPUs 0,1,4-7).
Usage: deploy_soft_t21.py SOFT_ROWS_T21NEW.jsonl.gz
The existing Luna soft rows are copied from the bucket (clause-labeling-20261006/soft-rows/); the new rows go up in ~2 MB chunks."""
import base64, hashlib, netrc, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent; CL = H.parents[3] / 'research/clause-labeling-20261005'
new = Path(sys.argv[1]); R = '/tmp/pangram-t2-20261006'
code_files = {n: (H / n).read_text() for n in ['run_soft_t21.sh', 'run_pair_soft.sh', 'soft_gate_build.py', 'score_worker_t21s.py']}
code_files['soften_prepared.py'] = (CL / 'soften_prepared.py').read_text()
run(f"""
import glob, shutil
from pathlib import Path
R = Path({R!r}); S = R / 'sweeps'
assert (R / 'runs/qwen35-4b-t21/prepared-v2').exists(), 'T2.1 prepared dir missing'
assert not list(S.glob('q4b-T21S-*')) and not (R / 'runs/qwen35-4b-t21soft').exists(), 'soft arm already exists'
for n in ['inputs/heldout-score-rows.jsonl.gz', 'inputs/t21-heldout-score-rows.jsonl.gz', 'inputs/t21-heldout-score-rows-v2-clean-for-t21.jsonl.gz',
          'inputs/polish-eval-v1.jsonl.gz', 'inputs/humanizer-eval-v1.jsonl.gz', 'inputs/cowrite-eval-v1.jsonl.gz', 'hetero', 'sweep_eval.py', 'cross_model_eval.py']:
    assert (R / n).exists(), n
src = glob.glob('/data/workspace/clause-labeling-20261006/soft-rows/soft-rows-existing.jsonl.gz')
assert src, 'existing soft rows not in bucket'; shutil.copy(src[0], R / 'inputs/soft-rows-existing.jsonl.gz')
(R / 'inputs/soft-rows-t21new.jsonl.gz.part').unlink(missing_ok=True); print('preflight ok')
""", timeout=120)
b = new.read_bytes(); step = 2_000_000
for i in range(0, len(b), step):
    run(f"open('{R}/inputs/soft-rows-t21new.jsonl.gz.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + step]).decode()!r}))", timeout=120)
run(f"""import os, hashlib, shutil
p = '{R}/inputs/soft-rows-t21new.jsonl.gz'; os.replace(p + '.part', p)
assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == {hashlib.sha256(b).hexdigest()!r}
shutil.copy(p, '/data/workspace/clause-labeling-20261006/soft-rows/soft-rows-t21new.jsonl.gz'); print('ok soft-rows-t21new')""", timeout=120)
start = r'''
import json, os, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'
for n, s in FILES.items(): (R / n).write_text(s)
env = os.environ.copy(); env['WANDB_API_KEY'] = WKEY
p = subprocess.Popen(['bash', str(R / 'run_soft_t21.sh'), sys.executable], cwd=R, env=env, stdout=open(S / 'run-soft-t21.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
(S / 'soft-t21-pids.json').write_text(json.dumps({'run_soft_t21_pid': p.pid})); print('started', p.pid)
'''
run('FILES=' + repr(code_files) + '\nWKEY=' + repr(netrc.netrc().authenticators('api.wandb.ai')[2]) + '\n' + start, timeout=120)
