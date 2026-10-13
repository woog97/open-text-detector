"""Per-GPU sweep queue: claim the next job in sweeps/queue.json, train it, then evaluate it if sweep_eval.py exists.

Jobs are claimed atomically (claim file under a lock). A failed job is re-queued once as <tag>-retry.
Edit sweeps/queue.json at any time to append jobs; it is re-read before each claim.
"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parent; S = R / 'sweeps'; gpu = sys.argv[1]
log = open(S / f'runner-gpu{gpu}.log', 'a')


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), 'gpu': gpu, **k}) + '\n'); log.flush()


def claim():
    with open(S / 'queue.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = json.loads((S / 'queue.json').read_text())
        for job in q:
            d = S / job['tag']
            if (d / 'claim.json').exists() or job.get('gpus') and gpu not in job['gpus']:
                continue
            d.mkdir(parents=True, exist_ok=True); (d / 'claim.json').write_text(json.dumps({'gpu': gpu, 'time': time.time()}))
            return job
    return None


def requeue(job):
    with open(S / 'queue.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = json.loads((S / 'queue.json').read_text())
        q.append({**job, 'tag': job['tag'] + '-retry', 'retry_of': job['tag']})
        (S / 'queue.json').write_text(json.dumps(q, indent=1))


env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                                   HF_HUB_DISABLE_PROGRESS_BARS='1', WANDB_DISABLE_CODE='true', WANDB_DISABLE_GIT='true', WANDB_CONSOLE='off')
if not env.get('WANDB_API_KEY'):
    say(event='abort', reason='WANDB_API_KEY missing'); sys.exit(1)
fast_failures = 0
while True:
    if fast_failures >= 2:
        (S / f'ALERT-gpu{gpu}').write_text('two consecutive fast failures; runner stopped'); say(event='circuit_breaker'); break
    if (S / 'STOP').exists():
        say(event='stop'); break
    job = claim()
    if job is None:
        say(event='queue_empty'); break
    d = S / job['tag']; say(event='train_start', tag=job['tag']); t0 = time.time()
    with open(d / 'train.log', 'a') as f:
        rc = subprocess.run([sys.executable, '-u', str(R / 'train_sweep.py'), job['model'], job['tag'], *job['args']],
                            env=env, cwd=R, stdout=f, stderr=f).returncode
    say(event='train_end', tag=job['tag'], rc=rc)
    if rc and (d / 'resume-state.pt').exists():
        say(event='resume_attempt', tag=job['tag'])
        with open(d / 'train.log', 'a') as f:
            rc = subprocess.run([sys.executable, '-u', str(R / 'train_sweep.py'), job['model'], job['tag'], *job['args'], '--resume'],
                                env=env, cwd=R, stdout=f, stderr=f).returncode
        say(event='resume_end', tag=job['tag'], rc=rc)
    if rc:
        fast_failures = fast_failures + 1 if time.time() - t0 < 900 else 0
        if not job.get('retry_of') and fast_failures < 2:
            requeue(job)
        continue
    fast_failures = 0
    if str(R).startswith('/tmp/') and (R / 'persist_daemon.py').exists():
        # Copy the finished run to /data right away; a Space restart wipes /tmp (Oct 5 and Oct 6 runs were lost this way).
        subprocess.run([sys.executable, str(R / 'persist_daemon.py'), str(R), '--once'], cwd=R)
        say(event='persisted', tag=job['tag'])
    if (R / 'sweep_eval.py').exists():
        # Evaluate in the background on the same GPU so the next training run starts immediately.
        f = open(d / 'eval.log', 'a')
        p = subprocess.Popen([sys.executable, '-u', str(R / 'sweep_eval.py'), job['tag']], env=env, cwd=R, stdout=f, stderr=f, start_new_session=True)
        say(event='eval_started', tag=job['tag'], pid=p.pid)
