"""Cross-model eval worker for one GPU: claims tags in order once their final checkpoint and sweep-eval scores exist.
Usage: cross_daemon.py GPU [WAIT_PID]   (WAIT_PID: start only after that process exits, e.g. the GPU's training runner)"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); S = R / 'sweeps'; gpu = sys.argv[1]; wait = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ORDER = [f'q4b-{a}-s{s}' for a in ('SPG', 'Arep', 'SPH', 'LLE', 'MIX') for s in (1, 2, 3)]
log = open(S / f'cross-gpu{gpu}.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()
while wait and os.path.exists(f'/proc/{wait}'):
    time.sleep(60)
say(event='start')
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
env['PYTHONPATH'] = '/tmp/pangram-wandb-vendor'
def ready(t):
    d = S / t
    return (d / 'eval/stage2-epoch2-sentences.npz').exists() and (d / 'stage2-epoch2-adapters.safetensors').exists() and \
        json.loads((d / 'status.json').read_text()).get('state') == 'trained'
def claim():
    with open(S / 'cross.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        pending = [t for t in ORDER if (S / t).exists() or t.startswith(('q4b-Arep', 'q4b-LLE', 'q4b-MIX'))]
        left = [t for t in pending if not (S / t / 'cross.claim').exists()]
        for t in left:
            if (S / t).exists() and ready(t):
                (S / t / 'cross.claim').write_text(json.dumps({'gpu': gpu, 't': time.time()})); return t, left
        return None, left
while True:
    t, left = claim()
    if t is None:
        if not left: say(event='all_done'); break
        time.sleep(120); continue
    d = S / t; t0 = time.time(); say(event='cross_start', tag=t)
    with open(d / 'cross.log', 'a') as f:
        rc = subprocess.run([sys.executable, '-u', str(R / 'cross_model_eval.py'), t, str(d / 'stage2-epoch2-adapters.safetensors'), str(R / 'hetero')],
                            cwd=R, env=env, stdout=f, stderr=f).returncode
    say(event='cross_end', tag=t, rc=rc, seconds=round(time.time() - t0))
