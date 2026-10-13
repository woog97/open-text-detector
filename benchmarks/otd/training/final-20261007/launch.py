"""Launch the final full-length single-epoch runs: Qwen3.5 4B (1 GPU), Qwen3.6 35B-A3B MoE (2 GPUs, layer split), EmbeddingGemma 2 (1 GPU).
No DDP anywhere. Run from the Mac. Steps:
  1. Fit the mix to 510 tokens under all three tokenizers on the Space (fit_rows.py) -> /data/workspace/final-20261007/mix/prepared-v2.
  2. Check free GPUs at launch time and place each run (free = no compute process and < 100 MiB used):
       MoE: 2 free Space A100s -> 2 free H200s -> 1 free H200 (141 GB holds the bf16 MoE).   4B, EmbeddingGemma: 1 Space A100 -> 1 H200.
     Space GPUs 2 and 3 are never used (reserved for Alice). The H200 is shared: only GPUs with nothing on them are taken.
  3. Start each run with run_one.sh (resume once after a crash). Space runs write /tmp/final-20261007/sweeps/<tag>, mirrored every
     5 minutes to /data/workspace/final-20261007/sweeps by persist_daemon. H200 runs write /workspace/woog/pangram/final-20261007/sweeps/<tag>
     (resume state on /dev/shm); copy them to the bucket afterwards with `launch.py --sync-h200`.
  4. Record the placement (host, GPUs, tag, args) in /data/workspace/final-20261007/placement.json and final-20261007/placements/.
Credentials: the W&B key (from ~/.netrc) goes to the Space inside the kernel request and to the H200 over ssh stdin; the HF token
(local huggingface_hub login) goes to the H200 over stdin only for bucket transfers. Nothing is written to disk.

  python launch.py --mix /data/workspace/<...>/prepared-v2 [--dry-run] [--only qwen35-4b,embeddinggemma-2] [--skip-fit]
  python launch.py --sync-h200          # upload finished H200 run folders to workspace/final-20261007/h200/ in the bucket
"""
import argparse, contextlib, io, json, netrc, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
import push as code_push

H = Path(__file__).resolve().parent
SR, SD = '/tmp/final-20261007', '/data/workspace/final-20261007'
HR = '/workspace/woog/pangram/final-20261007'
BUCKET = 'open-text-detector/training-storage'
COMMON = ['--fraction', '1.0', '--stage2-epochs', '3', '--head-lr', '2e-5', '--schedule', 'cosine', '--warmup', '0.06', '--seed', '1',
          '--resume-every', '20', '--snapshot-every', '100', '--snapshot-keep', '3', '--on-diverge', 'rollback']
RUNS = {  # model: (tag, preferred GPU count, args). MoE lr 1e-4: 2e-4 diverged on the H200 (moe-A-full); 1e-4 trained cleanly.
    'qwen36-35b-a3b': ('final-moe', 2, ['--micro-batch', '8', '--lr', '1e-4']),
    'qwen35-4b': ('final-qwen35-4b', 1, ['--micro-batch', '8', '--lr', '2e-4']),
    'embeddinggemma-2': ('final-eg2', 1, ['--micro-batch', '32', '--lr', '2e-4']),
}
RESERVED_SPACE = {2, 3}


def space(code, timeout=600):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(code, timeout=timeout)
    return buf.getvalue()


def h200(cmd, stdin=None, timeout=600, secrets=()):
    r = subprocess.run(['ssh', '-o', 'ConnectTimeout=20', 'otd-h200', cmd], input=stdin, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        err = r.stderr[-800:]
        for x in secrets: err = err.replace(x, '[REDACTED]')
        raise RuntimeError(f'h200 failed: {err}')
    return r.stdout


def h200_script(script, secrets, timeout):
    """Copy a script (no secrets in it) to the H200, then run it with the secrets as its stdin lines (read by its first lines)."""
    name = f'/tmp/woog-final-{os.getpid()}.sh'
    h200(f'umask 077; cat > {name}', stdin=script, timeout=60)
    try:
        return h200(f'bash {name}', stdin=''.join(x + '\n' for x in secrets), timeout=timeout, secrets=secrets)
    finally:
        h200(f'rm -f {name}', timeout=60)


FREE_GPUS = r'''
import subprocess, json
q = lambda a: subprocess.run(['nvidia-smi', a, '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip().splitlines()
gpus = [l.split(', ') for l in q('--query-gpu=index,uuid,memory.used')]
busy = {l.split(', ')[0] for l in q('--query-compute-apps=gpu_uuid,pid') if l.strip()}
print('FREE=' + json.dumps([int(i) for i, u, m in gpus if u not in busy and int(m) < 100]))
'''


def free_space():
    out = space(FREE_GPUS, timeout=120)
    return [g for g in json.loads(out.split('FREE=')[1].splitlines()[0]) if g not in RESERVED_SPACE]


def free_h200():
    try:
        out = h200('cat > /tmp/woog-free-gpus.py && python3 /tmp/woog-free-gpus.py; rm -f /tmp/woog-free-gpus.py', stdin=FREE_GPUS, timeout=60)
        return json.loads(out.split('FREE=')[1].splitlines()[0])
    except Exception as e:
        print('H200 unreachable:', repr(e)[:200]); return []


def place(models, sp, hp):
    """Placement order: MoE 2 Space -> 1 H200 (faster than a 2-H200 split: 3.7 vs 3.1 rows/s) -> 2 H200; others 1 Space -> 1 H200. The MoE is placed first; its first option in that
    order that still leaves room for every other run wins (else its first feasible option, and the others take what is left)."""
    def opts(m):
        return [('space', 2), ('h200', 1), ('h200', 2)] if RUNS[m][1] == 2 else [('space', 1), ('h200', 1)]

    def greedy(ms, pools):
        pools = {k: list(v) for k, v in pools.items()}; plan = {}
        for m in ms:
            plan[m] = None
            for host, n in opts(m):
                if len(pools[host]) >= n:
                    plan[m] = (host, [pools[host].pop(0) for _ in range(n)]); break
        return plan

    ms = sorted(models, key=lambda m: -RUNS[m][1]); pools = {'space': list(sp), 'h200': list(hp)}
    if not ms or RUNS[ms[0]][1] < 2: return greedy(ms, pools)
    first = None
    for host, n in opts(ms[0]):
        if len(pools[host]) < n: continue
        rest = {k: list(v) for k, v in pools.items()}; got = (host, [rest[host].pop(0) for _ in range(n)])
        plan = {ms[0]: got, **greedy(ms[1:], rest)}
        if all(plan.values()): return plan
        first = first or plan
    return first or greedy(ms, pools)


def args_for(m, host, gpus):
    tag, _, extra = RUNS[m]; a = COMMON + extra
    if len(gpus) > 1: a = a + ['--model-parallel']
    if host == 'h200': a = a + ['--state-dir', '/dev/shm/woog-pangram/state']
    return tag, a


def fit_mix(mix):
    """Fit the mix on the Space (detached, after the idempotent setup), wait for it, return the fit report."""
    code_push.push()
    print(space(f'''import subprocess, os
from pathlib import Path
D = Path({SD!r}); out = D / 'mix/prepared-v2'
pidf = D / 'fit.pid'
if (out / '.complete').exists(): print('fit already done')
elif pidf.exists() and Path('/proc/' + pidf.read_text().strip()).exists(): print('fit already running', pidf.read_text())
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    sh = f"python -u {SD}/code/space_setup.py > {SD}/setup.log 2>&1 && cd {SR} && PYTHONPATH={SR}/vendor HF_HUB_OFFLINE=1 python fit_rows.py {mix} /tmp/final-20261007/mix-fit {SR}/assets > {SD}/fit.log 2>&1 && rm -rf {{out}} && cp -r /tmp/final-20261007/mix-fit {{out}} && echo {mix} > {{out}}/source.txt && touch {{out}}/.complete"
    p = subprocess.Popen(['bash', '-c', sh], start_new_session=True); pidf.write_text(str(p.pid)); print('fit started', p.pid)''', timeout=120))
    while True:
        out = space(f'''from pathlib import Path
D = Path({SD!r}); p = D / 'mix/prepared-v2'; pidf = D / 'fit.pid'
alive = pidf.exists() and Path('/proc/' + pidf.read_text().strip()).exists()
tail = lambda f: (D / f).read_text()[-300:] if (D / f).exists() else ''
if (p / '.complete').exists(): print('DONE', (p / 'fit-report.json').read_text())
elif alive: print('WAIT', tail('fit.log') or tail('setup.log'))
else: print('FAILED Error', tail('setup.log'), tail('fit.log'))''', timeout=60)
        if out.startswith('DONE'): return out[5:]
        if 'Error' in out or 'Traceback' in out: raise RuntimeError('fit failed: ' + out[-600:])
        time.sleep(60)


def start_space(runs, wkey):
    """runs: [(model, tag, gpus, args)]. Re-runs setup (idempotent), links the fitted mix, starts persist_daemon and the runs."""
    code_push.push()
    return space('RUNS=' + repr(runs) + '\nWKEY=' + repr(wkey) + f'''
import json, os, shutil, subprocess, sys
from pathlib import Path
R, D = Path({SR!r}), Path({SD!r})
subprocess.run([sys.executable, '-u', str(D / 'code/space_setup.py')], cwd='/tmp', check=True, timeout=3000)
mix = R / 'mix/prepared-v2'
if not (mix / '.complete').exists():
    shutil.rmtree(mix, ignore_errors=True); shutil.copytree(D / 'mix/prepared-v2', mix)
for m, tag, gpus, args in RUNS:
    assert not (R / 'sweeps' / tag / 'run.json').exists(), f'{{tag}} already exists: not restarting (use --resume by hand)'
    (R / 'runs' / m).mkdir(parents=True, exist_ok=True); link = R / 'runs' / m / 'prepared-v2'
    if link.is_symlink() or link.exists(): link.unlink()
    link.symlink_to(mix)
pids = {{}}
pp = R / 'persist.pid'
if not (pp.exists() and Path('/proc/' + pp.read_text().strip()).exists()):
    p = subprocess.Popen([sys.executable, '-u', str(R / 'persist_daemon.py'), str(R), '--every', '300'], cwd=R, stdout=open(R / 'persist.out', 'a'),
                         stderr=subprocess.STDOUT, start_new_session=True); pp.write_text(str(p.pid)); pids['persist'] = p.pid
env = {{**os.environ, 'WANDB_API_KEY': WKEY}}
for m, tag, gpus, args in RUNS:
    p = subprocess.Popen(['bash', str(R / 'run_one.sh'), str(R), sys.executable, m, tag, ','.join(map(str, gpus)), *args], cwd=R, env=env,
                         stdout=open(R / f'{{tag}}.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True); pids[tag] = p.pid
print('PIDS=' + json.dumps(pids))''', timeout=3600)


def start_h200(runs, wkey, hf_token):
    for f in ['runtime.py', 'models.json', 'run_one.sh', 'h200_setup.sh', 'fit_rows.py', 'smoke_tokens.py']:
        subprocess.run(['scp', '-q', str(H / f), f'otd-h200:{HR}/'], check=True)
    for n, p in code_push.FILES.items():
        if n not in ('space_setup.py', 'persist_daemon.py'): subprocess.run(['scp', '-q', str(p), f'otd-h200:{HR}/{n}'], check=True)
    h200(f'cd {HR} && bash h200_setup.sh > setup.log 2>&1 && tail -1 setup.log', timeout=3000)
    script = f'''read -r HF_TOKEN; read -r WANDB_API_KEY; export WANDB_API_KEY
cd {HR}; mkdir -p mix
if [ ! -f mix/prepared-v2/.complete ]; then rm -rf mix/prepared-v2
  HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=0 venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN']); pre = 'workspace/final-20261007/mix/prepared-v2/'
files = [f for f in api.list_bucket_tree('{BUCKET}', prefix=pre, recursive=True) if hasattr(f, 'size')]
api.download_bucket_files('{BUCKET}', [(f, 'mix/prepared-v2/' + f.path[len(pre):]) for f in files], raise_on_missing_files=True)
print('mix files', len(files))
PY
fi
test -f mix/prepared-v2/.complete || {{ echo "mix download incomplete"; exit 1; }}
unset HF_TOKEN
'''
    for m, tag, gpus, args in runs:
        script += (f'test -f sweeps/{tag}/run.json && {{ echo "{tag} exists"; exit 1; }}\n'
                   f'mkdir -p runs/{m}; ln -sfn {HR}/mix/prepared-v2 runs/{m}/prepared-v2\n'
                   f'nohup bash run_one.sh {HR} venv/bin/python {m} {tag} {",".join(map(str, gpus))} {" ".join(args)} > {tag}.out 2>&1 &\n'
                   f'echo "started {tag} pid $!"\n')
    return h200_script(script, [hf_token, wkey], timeout=1800)


def sync_h200():
    from huggingface_hub import get_token
    script = f'''read -r HF_TOKEN; cd {HR}; HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=0 venv/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN']); R = Path('sweeps')
for d in sorted(p for p in R.glob('final-*') if p.is_dir()):
    files = [p for p in d.rglob('*') if p.is_file() and not p.name.endswith(('.tmp', '.pt')) and 'wandb' not in p.parts]
    for i in range(0, len(files), 200):
        api.batch_bucket_files('{BUCKET}', add=[(str(p), 'workspace/final-20261007/h200/' + str(p.relative_to(R))) for p in files[i:i + 200]])
    print(d.name, len(files), 'files uploaded')
PY'''
    print(h200_script(script, [get_token()], timeout=3600))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mix'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--only')
    ap.add_argument('--skip-fit', action='store_true'); ap.add_argument('--sync-h200', action='store_true')
    a = ap.parse_args()
    if a.sync_h200: return sync_h200()
    models = a.only.split(',') if a.only else list(RUNS)
    if not a.skip_fit and not a.dry_run:
        assert a.mix, '--mix PREPARED_DIR (on the Space) is required'
        print('fit report:', fit_mix(a.mix))
    sp, hp = free_space(), free_h200()
    plan = place(models, sp, hp)
    print(json.dumps({'space_free': sp, 'h200_free': hp, 'plan': plan}))
    if a.dry_run: return
    missing = [m for m, v in plan.items() if v is None]
    if missing: print('NOT PLACED (no free GPUs):', missing)
    wkey = netrc.netrc().authenticators('api.wandb.ai')[2]
    record = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'mix': a.mix, 'runs': {}, 'not_placed': missing}
    sruns = [(m, *args_for(m, 'space', g)[:1], g, args_for(m, 'space', g)[1]) for m, v in plan.items() if v and v[0] == 'space' for g in [v[1]]]
    hruns = [(m, *args_for(m, 'h200', g)[:1], g, args_for(m, 'h200', g)[1]) for m, v in plan.items() if v and v[0] == 'h200' for g in [v[1]]]
    if sruns: print(start_space(sruns, wkey)[-600:])
    if hruns:
        from huggingface_hub import get_token
        print(start_h200(hruns, wkey, get_token())[-600:])
    for host, rs in (('space', sruns), ('h200', hruns)):
        for m, tag, g, args in rs:
            record['runs'][tag] = {'model': m, 'host': host, 'gpus': g, 'root': SR if host == 'space' else HR, 'args': args}
    (H / 'placements').mkdir(exist_ok=True)
    (H / 'placements' / f"placement-{record['started_utc'].replace(':', '')}.json").write_text(json.dumps(record, indent=1))
    space(f"open({SD!r} + '/placement.json', 'w').write({json.dumps(record, indent=1)!r}); print('recorded')", timeout=60)
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
