"""Soft-arm (q4b-T21S-*) scoring worker for one GPU: the same evaluations the T2.1 seeds got, so the arms pair up.
Steps per run: sweep eval -> old held-out writers, T2.1 held-out v1, clean v2, cross-model s500, three Fable sets.
Usage: score_worker_t21s.py GPU"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'; gpu = sys.argv[1]
log = open(S / f'score3-gpu{gpu}.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=f'{R}/vendor:/tmp/pangram-wandb-vendor')
TAGS = [f'q4b-T21S-s{s}' for s in (1, 2, 3)]; CK = 'stage2-epoch2'
SETS = {'heldout': ('heldout-score-rows.jsonl.gz', 'heldout-writers'), 't21': ('t21-heldout-score-rows.jsonl.gz', 't21-heldout'),
        't21v2': ('t21-heldout-score-rows-v2-clean-for-t21.jsonl.gz', 't21-heldout-v2-clean'), 'polish': ('polish-eval-v1.jsonl.gz', 'fable-polish'),
        'humanizer': ('humanizer-eval-v1.jsonl.gz', 'fable-humanizer'), 'cowrite': ('cowrite-eval-v1.jsonl.gz', 'fable-cowrite')}
TASKS = [(t, 'sweep') for t in TAGS] + [(t, k) for t in TAGS for k in ('t21v2', 'heldout', 'cross', 'polish', 'humanizer', 'cowrite', 't21')]
def ready(t, step):
    d = S / t
    if step == 'sweep': return (d / 'train.done').exists() and (d / f'{CK}-adapters.safetensors').exists()
    return (d / 'eval' / f'{CK}-sentences.npz').exists() and (d / f'{CK}-adapters.safetensors').exists()
def cmd(t, step):
    if step == 'sweep': return [sys.executable, '-u', str(R / 'sweep_eval.py'), t]
    ck = str(S / t / f'{CK}-adapters.safetensors')
    if step == 'cross': return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'hetero'), 'cross-model-s500', '500']
    f, name = SETS[step]; return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'inputs' / f), name]
def claim():
    with open(S / 'score.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        left = [x for x in TASKS if not (S / x[0] / f'score3-{x[1]}.claim').exists()]
        for t, step in left:
            if (S / t).exists() and ready(t, step):
                (S / t / f'score3-{step}.claim').write_text(json.dumps({'gpu': gpu, 't': time.time()})); return (t, step), left
        return None, left
say(event='start')
while True:
    job, left = claim()
    if job is None:
        if not left: say(event='all_claimed'); break
        time.sleep(60); continue
    t, step = job; t0 = time.time(); say(event='begin', tag=t, step=step)
    with open(S / t / f'score3-{step}.log', 'a') as f:
        rc = subprocess.run(cmd(t, step), cwd=R, env=env, stdout=f, stderr=f).returncode
    say(event='end', tag=t, step=step, rc=rc, seconds=round(time.time() - t0))
    if rc: (S / t / f'score3-{step}.failed').write_text(str(rc))
