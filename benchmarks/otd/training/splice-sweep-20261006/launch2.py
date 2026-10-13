import json, netrc, os, sys
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
A = ['--fraction', '0.2', '--micro-batch', '8', '--lr', '2e-4', '--head-lr', '2e-5', '--schedule', 'cosine', '--warmup', '0.06']
q = []
for arm, model, gpus in [('LLE', 'qwen35-4b-lle', ['0', '1', '2']), ('MIX', 'qwen35-4b-mix', ['3', '4', '5']), ('Arep', 'qwen35-4b', ['6', '7'])]:
    for s in (1, 2, 3):
        q.append({'tag': f'q4b-{arm}-s{s}', 'model': model, 'arm': arm, 'seed': s, 'args': A + ['--seed', str(s)], 'gpus': gpus})
code = r'''
import json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); S = R / 'sweeps'
models = {m['name'] for m in json.loads((R / 'models.json').read_text())}
missing = [j['model'] for j in Q if j['model'] not in models] + [f for f in ['cross_daemon.py', 'cross_model_eval.py', 'sweep_eval.py', 'queue_runner.py'] if not (R / f).exists()]
assert not missing, missing
old = json.loads((S / 'queue.json').read_text()) if (S / 'queue.json').exists() else []
(S / 'queue.json').write_text(json.dumps(old + [j for j in Q if j['tag'] not in {o['tag'] for o in old}], indent=1))
env = os.environ.copy(); env['WANDB_API_KEY'] = WKEY
pids = {}
for g in [str(i) for i in range(8)]:
    p = subprocess.Popen([sys.executable, '-u', str(R / 'queue_runner.py'), g], cwd=R, env=env, stdout=open(S / f'runner-gpu{g}.out', 'a'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    pids[g] = p.pid
env2 = {k: v for k, v in os.environ.items() if k != 'WANDB_API_KEY'}
cross = {}
for g, pid in pids.items():
    c = subprocess.Popen([sys.executable, '-u', str(R / 'cross_daemon.py'), g, str(pid)], cwd=R, env=env2, stdout=open(S / f'cross-gpu{g}.out', 'a'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    cross[g] = c.pid
(S / 'runner-pids-wave2.json').write_text(json.dumps({'runners': pids, 'cross': cross})); print(json.dumps({'runners': pids, 'cross': cross}))
'''
run('Q=' + repr(q) + '\nWKEY=' + repr(netrc.netrc().authenticators('api.wandb.ai')[2]) + '\n' + code, timeout=120)
