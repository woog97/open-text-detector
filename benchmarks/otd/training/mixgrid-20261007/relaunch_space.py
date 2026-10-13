"""Idempotent (re)launch of the full-size mixA run on the Space after a restart or /tmp wipe (Oct 8).

Run on the Space with WANDB_API_KEY in the environment (only needed when it actually starts training):
  python relaunch_space.py [--setup-only]
1. /tmp/final-20261007 (vendor + MoE weights): rebuilt by its own space_setup.py when missing.
2. /tmp/mixfull-20261008: rebuilt from /data/workspace/mixfull-20261008/code.tgz; links to the vendor, the weights and the fitted
   mix on /data (so the data never depends on /tmp).
3. Sweep outputs restored from /data with persist_daemon --restore; a persist_daemon is kept running.
4. Unless train.done exists or the run is alive, run_fsdp_full.sh is started on the GPUs in launch.json (each must be free);
   it resumes from the step-tagged state on /data. Prints one JSON status line.
"""
import json, os, subprocess, sys, tarfile, time
from pathlib import Path

D = Path('/data/workspace/mixfull-20261008'); R = Path('/tmp/mixfull-20261008'); F = Path('/tmp/final-20261007')
cfg = json.loads((D / 'launch.json').read_text())  # {"model", "tag", "gpus", "port", "data"}
out = {'t': time.strftime('%H:%M:%S')}


def alive_pid(f):
    """Recorded pid (file f) is a live, non-zombie process."""
    try:
        pid = f.read_text().strip(); st = Path(f'/proc/{pid}/stat')
        return st.exists() and st.read_text().split()[2] != 'Z'
    except Exception:
        return False


def free(g):
    q = lambda a: subprocess.run(['nvidia-smi', a, '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip().splitlines()
    busy = {l.split(', ')[0] for l in q('--query-compute-apps=gpu_uuid,pid') if l.strip()}
    return any(i == str(g) and u not in busy and int(m) < 100 for i, u, m in (l.split(', ') for l in q('--query-gpu=index,uuid,memory.used')))


if not (F / 'assets/qwen36-35b-a3b/.complete').exists() or not (F / 'vendor').exists():
    subprocess.run([sys.executable, '-u', '/data/workspace/final-20261007/code/space_setup.py'], cwd='/tmp', check=True, timeout=3600)
    out['space_setup'] = 'ran'
if not (R / 'train_sweep.py').exists():
    R.mkdir(exist_ok=True)
    with tarfile.open(D / 'code.tgz') as t: t.extractall(R)
    out['code'] = 'restored'
links = {R / 'vendor': F / 'vendor', R / 'assets' / cfg['model']: F / 'assets/qwen36-35b-a3b',
         R / 'runs' / cfg['model'] / 'prepared-v2': Path(cfg['data'])}
for l, target in links.items():
    l.parent.mkdir(parents=True, exist_ok=True)
    if not l.exists() and not l.is_symlink(): l.symlink_to(target)
if not (R / 'sweeps' / cfg['tag']).exists() and (D / 'sweeps' / cfg['tag']).exists():
    subprocess.run([sys.executable, 'persist_daemon.py', str(R), '--restore'], cwd=R, check=True, timeout=1800); out['sweeps'] = 'restored'
if not alive_pid(D / 'persist.pid'):
    pd = subprocess.Popen([sys.executable, '-u', 'persist_daemon.py', str(R), '--every', '300'], cwd=R, stdout=open(D / 'persist.out', 'a'),
                          stderr=subprocess.STDOUT, start_new_session=True)
    (D / 'persist.pid').write_text(str(pd.pid)); out['persist_daemon'] = 'started'
done = R / 'sweeps' / cfg['tag'] / 'train.done'
try: state = json.loads((R / 'sweeps' / cfg['tag'] / 'status.json').read_text())['state']
except Exception: state = None
if '--setup-only' in sys.argv: out['run'] = 'setup only'
elif done.exists() and done.read_text().startswith('rc=0') and state == 'trained': out['run'] = 'finished'
elif state == 'diverged': out['run'] = 'diverged: not restarting'
elif alive_pid(D / 'wrapper.pid'): out['run'] = 'alive'
elif not (os.environ.get('WANDB_API_KEY') or os.environ.get('WANDB_MODE') == 'offline'): out['run'] = 'refusing: empty W&B key'
elif not all(free(g) for g in cfg['gpus'].split(',')): out['run'] = 'GPUs busy (retry later): ' + cfg['gpus']
else:
    done.unlink(missing_ok=True)
    p = subprocess.Popen(['bash', str(R / 'run_fsdp_full.sh'), cfg['model'], cfg['tag'], cfg['gpus'], str(cfg['port'])], cwd=R,
                         stdout=open(D / f"{cfg['tag']}-wrapper.out", 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    (D / 'wrapper.pid').write_text(str(p.pid)); out['run'] = f'started pid {p.pid}'
print(json.dumps(out))
