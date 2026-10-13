"""T2.1 scoring worker for one GPU: claims (run, step) tasks whose inputs exist. Steps per run: sweep eval (T2 only) ->
held-out writer eval -> cross-model eval. Usage: score_worker.py GPU [WAIT_FILE]  (start after WAIT_FILE exists)."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'; gpu = sys.argv[1]; wait = sys.argv[2] if len(sys.argv) > 2 else None
log = open(S / f'score-gpu{gpu}.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()
while wait and not Path(wait).exists(): time.sleep(30)
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=f'{R}/vendor:/tmp/pangram-wandb-vendor')
T21 = [f'q4b-T21-s{s}' for s in (1, 2, 3)]; REF = [f'q4b-T2-s{s}' for s in (1, 2, 3)] + [f'q4b-SPG-s{s}' for s in (1, 2, 3)]
CK = 'stage2-epoch2'
# T2.1 seeds: sweep eval, old held-out writers, new T2.1 held-out set, cross-model. T2/SPG get the new set only, as a
# reference (T2 trained on part of it: Opus/Sonnet slice and GPT-6 Sol; SPG may share host papers via splices).
TASKS = [(t, 'sweep') for t in T21] + [(t, s) for t in T21 for s in ('t21', 'heldout', 'cross')] + [(t, 't21') for t in REF]
def ready(t, step):
    d = S / t
    if step == 'sweep':
        return (d / 'train.done').exists() and (d / f'{CK}-adapters.safetensors').exists()
    return (d / 'eval' / f'{CK}-sentences.npz').exists() and (d / f'{CK}-adapters.safetensors').exists()
def cmd(t, step):
    if step == 'sweep': return [sys.executable, '-u', str(R / 'sweep_eval.py'), t]
    ck = str(S / t / f'{CK}-adapters.safetensors')
    if step == 't21': return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'inputs/t21-heldout-score-rows.jsonl.gz'), 't21-heldout']
    if step == 'heldout': return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'inputs/heldout-score-rows.jsonl.gz'), 'heldout-writers']
    return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'hetero'), 'cross-model-s500', '500']
def claim():
    with open(S / 'score.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        left = [x for x in TASKS if not (S / x[0] / f'score-{x[1]}.claim').exists()]
        for t, step in left:
            if (S / t).exists() and ready(t, step):
                (S / t / f'score-{step}.claim').write_text(json.dumps({'gpu': gpu, 't': time.time()})); return (t, step), left
        return None, left
say(event='start')
while True:
    job, left = claim()
    if job is None:
        if not left: say(event='all_claimed'); break
        time.sleep(60); continue
    t, step = job; t0 = time.time(); say(event='begin', tag=t, step=step)
    with open(S / t / f'score-{step}.log', 'a') as f:
        rc = subprocess.run(cmd(t, step), cwd=R, env=env, stdout=f, stderr=f).returncode
    say(event='end', tag=t, step=step, rc=rc, seconds=round(time.time() - t0))
    if rc: (S / t / f'score-{step}.failed').write_text(str(rc))
