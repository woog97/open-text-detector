"""Mix-grid MoE scoring worker for one GPU (claims tasks under a shared lock). Paper-edit set on stage-2 epochs 2, 0, 1
(epoch 2 first: the other sets need its sentence scores for the paper threshold), then on epoch 2: clean v2, strict
held-out writers, T2.1 held-out v1, cross-model s500, Fable polish / humanizer / co-writing. Each finished task is copied
to /data. Usage: score_worker_mg.py GPU [WAIT_TAG]  (WAIT_TAG: wait for that run's train.done and a free GPU first)."""
import fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/mixgrid-20261007'); S = R / 'sweeps'; D = Path('/data/workspace/mixgrid-20261007/sweeps'); T2 = Path('/tmp/pangram-t2-20261006')
gpu = sys.argv[1]; wait = sys.argv[2] if len(sys.argv) > 2 else None
log = open(S / f'score-mg-gpu{gpu}.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()
def gpu_free():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return any(l.split(',')[0].strip() == gpu and int(l.split(',')[1]) < 1000 for l in out.strip().splitlines())
if wait:
    say(event='waiting', for_tag=wait)
    while not ((S / wait / 'train.done').exists() and gpu_free()): time.sleep(60)
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH='/tmp/final-20261007/vendor',
                                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
SETS = {'t21v2': (T2 / 'inputs/t21-heldout-score-rows-v2-clean-for-t21.jsonl.gz', 't21-heldout-v2-clean', ''), 'heldout': (T2 / 'inputs/heldout-score-rows.jsonl.gz', 'heldout-writers', ''),
        't21': (T2 / 'inputs/t21-heldout-score-rows.jsonl.gz', 't21-heldout', ''), 'cross': (T2 / 'hetero', 'cross-model-s500', '500'),
        'polish': (T2 / 'inputs/polish-eval-v1.jsonl.gz', 'fable-polish', ''), 'humanizer': (T2 / 'inputs/humanizer-eval-v1.jsonl.gz', 'fable-humanizer', ''),
        'cowrite': (T2 / 'inputs/cowrite-eval-v1.jsonl.gz', 'fable-cowrite', '')}
TAGS = ['moe-mixB', 'moe-mixA']
TASKS = [(t, s) for t in TAGS for s in ['sweep-e2', 't21v2', 'heldout', 'cross', 'humanizer', 'sweep-e0', 'sweep-e1', 't21', 'polish', 'cowrite']]
def ready(t, s):
    d = S / t
    if not (d / 'train.done').exists(): return False
    return True if s.startswith('sweep') else (d / 'eval/stage2-epoch2.json').exists()
def cmd(t, s):
    if s.startswith('sweep'): return [sys.executable, '-u', str(R / 'sweep_one.py'), t, f'stage2-epoch{s[-1]}']
    f, name, n = SETS[s]; c = [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, str(S / t / 'stage2-epoch2-adapters.safetensors'), str(f), name]
    return c + ([n] if n else [])
def claim():
    with open(S / 'score-mg.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        left = [x for x in TASKS if not (S / x[0] / f'mg-{x[1]}.claim').exists()]
        for t, s in left:
            if ready(t, s):
                (S / t / f'mg-{s}.claim').write_text(json.dumps({'gpu': gpu, 't': time.time()})); return (t, s), left
        return None, left
say(event='start')
while True:
    job, left = claim()
    if job is None:
        if not left: say(event='all_claimed'); break
        time.sleep(30); continue
    t, s = job; t0 = time.time(); say(event='begin', tag=t, step=s)
    with open(S / t / f'mg-{s}.log', 'a') as f:
        rc = subprocess.run(cmd(t, s), cwd=R, env=env, stdout=f, stderr=f).returncode
    say(event='end', tag=t, step=s, rc=rc, seconds=round(time.time() - t0))
    if rc: (S / t / f'mg-{s}.failed').write_text(str(rc))
    else:
        (D / t / 'eval').mkdir(parents=True, exist_ok=True)
        for p in (S / t / 'eval').iterdir():
            q = D / t / 'eval' / p.name
            if not q.exists() or q.stat().st_size != p.stat().st_size: shutil.copy(p, q)
