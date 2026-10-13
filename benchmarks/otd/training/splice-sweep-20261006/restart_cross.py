import json, os, signal, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); S = R / 'sweeps'
pids = json.loads((S / 'runner-pids-wave2.json').read_text())
new = {}
env = {k: v for k, v in os.environ.items() if k != 'WANDB_API_KEY'}
for g in ['0', '1', '4', '5', '7']:
    try: os.killpg(pids['cross'][g], signal.SIGKILL)
    except ProcessLookupError: pass
    c = subprocess.Popen([sys.executable, '-u', str(R / 'cross_daemon.py'), g], cwd=R, env=env, stdout=open(S / f'cross-gpu{g}.out', 'a'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    new[g] = c.pid; pids['cross'][g] = c.pid
(S / 'runner-pids-wave2.json').write_text(json.dumps(pids))
time.sleep(90)
print(json.dumps(new))
for g in new:
    f = S / f'cross-gpu{g}.log'; print(g, f.read_text().splitlines()[-2:] if f.exists() else None)
print(subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout)
