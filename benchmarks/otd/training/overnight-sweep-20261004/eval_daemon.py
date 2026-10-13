"""Background evaluation scheduler: score finished sweep runs (and growing learning-curve runs) on their own GPU.

Runs sweep_eval_run.py once per finished run, or with --watch for runs that save step checkpoints while
training. Never starts more than one evaluation per run. Stops when sweeps/STOP-EVAL exists.
"""
import json, os, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parent; S = R / 'sweeps'
log = open(S / 'eval-daemon.log', 'a')


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()


def alive(pid):
    return Path(f'/proc/{pid}').exists()


while not (S / 'STOP-EVAL').exists():
    for d in sorted(p for p in S.iterdir() if p.is_dir() and not p.name.startswith(('_', 'smoke', 'evaltest'))):
        claim, status = d / 'claim.json', d / 'status.json'
        if not claim.exists() or not status.exists() or not (d / 'run.json').exists():
            continue
        proc = d / 'eval-process.json'
        if proc.exists() and alive(json.loads(proc.read_text())['pid']):
            continue
        state = json.loads(status.read_text()).get('state')
        ckpts = [p for p in d.glob('*-adapters.safetensors') if not p.name.startswith('stage1')]
        done = {p.stem for p in (d / 'eval').glob('*.json')} if (d / 'eval').exists() else set()
        pending = [p for p in ckpts if p.name.split('-adapters')[0] not in done]
        if not pending:
            continue
        growing = state not in ('trained', 'failed') and any(p.name.startswith('step') for p in ckpts)
        if state not in ('trained', 'failed') and not growing:
            continue
        gpu = json.loads(claim.read_text())['gpu']
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        env.pop('WANDB_API_KEY', None)
        f = open(d / 'eval.log', 'a')
        p = subprocess.Popen([sys.executable, '-u', str(R / 'sweep_eval_run.py'), d.name] + (['--watch'] if growing else []),
                             env=env, cwd=R, stdout=f, stderr=f, start_new_session=True)
        proc.write_text(json.dumps({'pid': p.pid, 'gpu': gpu, 'watch': growing})); say(event='eval_started', tag=d.name, gpu=gpu, pending=len(pending), watch=growing)
    time.sleep(60)
say(event='stopped')
