import json, netrc, os, sys
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
A = ['--fraction', '0.2', '--micro-batch', '8', '--lr', '2e-4', '--head-lr', '2e-5', '--schedule', 'cosine', '--warmup', '0.06']
q = []
for arm, model, gpus in [('SPH', 'qwen35-4b-sph', ['0', '1', '2']), ('SPG', 'qwen35-4b-spg', ['3', '4', '5'])]:
    for s in (1, 2, 3):
        q.append({'tag': f'q4b-{arm}-s{s}', 'model': model, 'arm': arm, 'seed': s, 'args': A + ['--seed', str(s)], 'gpus': gpus})
code = r'''
import json, os, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); S = R / 'sweeps'
(S / 'queue.json').write_text(json.dumps(Q, indent=1))
env = os.environ.copy(); env['WANDB_API_KEY'] = WKEY
pids = {}
for g in ['0', '1', '2', '3', '4', '5']:
    p = subprocess.Popen([sys.executable, '-u', str(R / 'queue_runner.py'), g], cwd=R, env=env, stdout=open(S / f'runner-gpu{g}.out', 'a'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    pids[g] = p.pid
(S / 'runner-pids.json').write_text(json.dumps(pids)); print(json.dumps(pids))
'''
run('Q=' + repr(q) + '\nWKEY=' + repr(netrc.netrc().authenticators('api.wandb.ai')[2]) + '\n' + code, timeout=120)
