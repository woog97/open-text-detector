"""Space side, soft arm: wait until T2.1 scoring has left GPUs 0,1,4-7 (rounds 1-2 finished, memory free twice in a row),
then build runs/qwen35-4b-t21soft/prepared-v2 from the T2.1 prepared dir with soft_prepared.py and persist it to /data.
Never touches GPUs 2-3. Writes progress to sweeps/soft-t21.log; exits non-zero on failure."""
import json, os, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'; D = Path('/data/workspace/pangram-t2-20261006'); log = open(S / 'soft-t21.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()
def alive(p): return os.path.exists(f'/proc/{p}')
GPUS = (0, 1, 4, 5, 6, 7); free_twice = 0
say(event='gate_wait')
while True:
    pids = [json.loads((S / 't21-pids.json').read_text())['run_t21_pid']] + list(json.loads((S / 't21b-pids.json').read_text()).values())
    used = {int(l.split(',')[0]): int(l.split(',')[1]) for l in subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip().splitlines()}
    ok = not any(alive(p) for p in pids) and all(used[g] < 1000 for g in GPUS)
    free_twice = free_twice + 1 if ok else 0
    if free_twice >= 2: break
    time.sleep(60)
say(event='gate_open')
src, out = R / 'runs/qwen35-4b-t21/prepared-v2', R / 'runs/qwen35-4b-t21soft/prepared-v2'
if not out.exists():
    rc = subprocess.run([sys.executable, '-u', str(R / 'soften_prepared.py'), str(src), str(out), str(R / 'inputs/soft-rows-existing.jsonl.gz'),
                         str(R / 'inputs/soft-rows-t21new.jsonl.gz')], stdout=open(S / 'soft-build.out', 'a'), stderr=subprocess.STDOUT).returncode
    if rc: say(event='build_failed', rc=rc); sys.exit(1)
models = json.loads((R / 'models.json').read_text())
if not any(m['name'] == 'qwen35-4b-t21soft' for m in models):
    models.append({**next(m for m in models if m['name'] == 'qwen35-4b-t21'), 'name': 'qwen35-4b-t21soft'}); (R / 'models.json').write_text(json.dumps(models, indent=1))
if not (R / 'assets/qwen35-4b-t21soft').exists(): (R / 'assets/qwen35-4b-t21soft').symlink_to(R / 'assets/qwen35-4b')
import shutil
shutil.copytree(out, D / 'prepared/qwen35-4b-t21soft/prepared-v2', dirs_exist_ok=True)
say(event='built', stats=json.loads((out / 'soften-stats.json').read_text()))
