"""T2.1 second-round scoring worker (v2 split + Fable polish/humanizer/co-writing) for one GPU: claims (run, step) tasks whose inputs exist. Steps per run: sweep eval (T2 only) ->
held-out writer eval -> cross-model eval. Usage: score_worker.py GPU [WAIT_FILE]  (start after WAIT_FILE exists)."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'; gpu = sys.argv[1]; wait = sys.argv[2] if len(sys.argv) > 2 else None
log = open(S / f'score2-gpu{gpu}.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()
while wait and not Path(wait).exists(): time.sleep(30)
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=f'{R}/vendor:/tmp/pangram-wandb-vendor')
T21 = [f'q4b-T21-s{s}' for s in (1, 2, 3)]; REF = [f'q4b-T2-s{s}' for s in (1, 2, 3)] + [f'q4b-SPG-s{s}' for s in (1, 2, 3)]
CK = 'stage2-epoch2'
# Second round (v2 split + Fable sets). T2.1 is scored on the v2 rows minus papers it may have trained on; T2/SPG get the
# full v2 file as a reference (T2 trained on part of it). All models get the three Fable sets (Fable is in no training mix).
SETS = {'t21v2': 't21-heldout-score-rows-v2-clean-for-t21.jsonl.gz', 't21v2full': 't21-heldout-score-rows-v2.jsonl.gz',
        'polish': 'polish-eval-v1.jsonl.gz', 'humanizer': 'humanizer-eval-v1.jsonl.gz', 'cowrite': 'cowrite-eval-v1.jsonl.gz'}
NAMES = {'t21v2': 't21-heldout-v2-clean', 't21v2full': 't21-heldout-v2', 'polish': 'fable-polish', 'humanizer': 'fable-humanizer', 'cowrite': 'fable-cowrite'}
TASKS = [(t, k) for t in T21 for k in ('t21v2', 'polish', 'humanizer', 'cowrite')] + [(t, k) for t in REF for k in ('t21v2full', 'polish', 'humanizer', 'cowrite')]
def ready(t, step):
    d = S / t
    if step == 'sweep':
        return (d / 'train.done').exists() and (d / f'{CK}-adapters.safetensors').exists()
    return (d / 'eval' / f'{CK}-sentences.npz').exists() and (d / f'{CK}-adapters.safetensors').exists()
def cmd(t, step):
    ck = str(S / t / f'{CK}-adapters.safetensors')
    return [sys.executable, '-u', str(R / 'cross_model_eval.py'), t, ck, str(R / 'inputs' / SETS[step]), NAMES[step]]
def claim():
    with open(S / 'score.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        left = [x for x in TASKS if not (S / x[0] / f'score2-{x[1]}.claim').exists()]
        for t, step in left:
            if (S / t).exists() and ready(t, step):
                (S / t / f'score2-{step}.claim').write_text(json.dumps({'gpu': gpu, 't': time.time()})); return (t, step), left
        return None, left
say(event='start')
while True:
    job, left = claim()
    if job is None:
        if not left: say(event='all_claimed'); break
        time.sleep(60); continue
    t, step = job; t0 = time.time(); say(event='begin', tag=t, step=step)
    with open(S / t / f'score2-{step}.log', 'a') as f:
        rc = subprocess.run(cmd(t, step), cwd=R, env=env, stdout=f, stderr=f).returncode
    say(event='end', tag=t, step=step, rc=rc, seconds=round(time.time() - t0))
    if rc: (S / t / f'score2-{step}.failed').write_text(str(rc))
