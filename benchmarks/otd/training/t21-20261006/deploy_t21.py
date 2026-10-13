"""Push the T2.1 split, builder and job chain to the Space and start run_t21.sh. Data files go in ~2 MB base64 chunks."""
import base64, json, netrc, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent; RS = H.parents[3] / 'research/t21-heldout-20261006'; SW = H.parent / 'overnight-sweep-20261004'
code_files = {n: (H / n).read_text() for n in ['build_t21.py', 'nt21.py', 'run_pair_t21.sh', 'score_worker_t21.py', 'run_t21.sh', 'persist_t21_build.py']}
code_files['cross_model_eval.py'] = (SW / 'cross_model_eval.py').read_text()
pre = r'''
import json, os, subprocess
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'
assert R.exists() and (R / 'runs/qwen35-4b/prepared-v2').exists() and (R / 'inputs/t2-pools.jsonl.gz').exists(), 'T2 workspace missing'
assert not list(S.glob('q4b-T21-*')), 'T2.1 runs already exist'
g = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
busy = [l for l in g.strip().splitlines() if int(l.split(',')[1]) > 1000]; assert not busy, busy
for f in Path(R / 'inputs').glob('t21-*.part'): f.unlink()
print('preflight ok')
'''
run(pre, timeout=60)
for name in ['t21-heldout-never-train.json', 't21-heldout-score-rows.jsonl.gz']:
    b = (RS / name).read_bytes(); step = 2_000_000
    for i in range(0, len(b), step):
        run(f"open('/tmp/pangram-t2-20261006/inputs/{name}.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + step]).decode()!r}))", timeout=120)
    run(f"""import os, hashlib
p='/tmp/pangram-t2-20261006/inputs/{name}'; os.replace(p + '.part', p)
assert hashlib.sha256(open(p,'rb').read()).hexdigest() == {__import__('hashlib').sha256(b).hexdigest()!r}; print('ok {name}')""", timeout=60)
start = r'''
import json, os, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'
for n, s in FILES.items(): (R / n).write_text(s)
pids = json.loads((S / 'job-pids.json').read_text()) if (S / 'job-pids.json').exists() else {}
out = {'persist_alive': bool(pids.get('persist')) and Path(f"/proc/{pids['persist']}").exists()}
if not out['persist_alive']:
    out['persist_pid'] = subprocess.Popen([sys.executable, '-u', str(R / 'persist_daemon.py'), str(R), '--every', '300'], cwd=R,
        stdout=open(S / 'persist.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True).pid
env = os.environ.copy(); env['WANDB_API_KEY'] = WKEY
p = subprocess.Popen(['bash', str(R / 'run_t21.sh'), sys.executable], cwd=R, env=env, stdout=open(S / 'run-t21.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
out['run_t21_pid'] = p.pid; (S / 't21-pids.json').write_text(json.dumps(out)); print(json.dumps(out))
'''
run('FILES=' + repr(code_files) + '\nWKEY=' + repr(netrc.netrc().authenticators('api.wandb.ai')[2]) + '\n' + start, timeout=120)
