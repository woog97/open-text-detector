"""Mac-side watcher for the full-size mix runs (Oct 8): mixA-full on the Space, mixB-full on the H200.

Checks at 1, 2, 4, 8, 16 and 32 minutes after start, then hourly, until both runs report trained (or diverged):
- host reboots (Space and H200 boot time; the Space also wipes /tmp), dead wrappers, stalled progress (logical_rows unchanged
  between two checks), the H200 /workspace quota (2000 GB minus `du`) and "Disk quota" lines in the H200 train.log;
- on a reboot or a dead unfinished run it runs that host's idempotent relaunch script (relaunch_space.py / relaunch_h200.sh).
Credentials: the W&B key from ~/.netrc and the HF token from huggingface_hub travel only in the request body / ssh stdin.
Log: --log (default ./mixfull-watch.log), one JSON line per check. Run: nohup python mixfull_watch.py --log FILE &
"""
import argparse, json, netrc, os, re, subprocess, sys, time
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'final-20261007'))
from remote import run  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--log', default='mixfull-watch.log'); a = ap.parse_args()
log = open(a.log, 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()
REDACT = re.compile(r'(hf_[A-Za-z0-9]+|wandb' r'_v1_[A-Za-z0-9_]+|eyJ[A-Za-z0-9._-]{20,}|token=[A-Za-z0-9_]+)')  # split so the pre-commit secret scan doesn't match the pattern itself


def space(code, timeout=300):
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): run(code, timeout=timeout)
    return REDACT.sub('[REDACTED]', buf.getvalue())


def h200(cmd, stdin=None, timeout=300):
    r = subprocess.run(['ssh', '-o', 'ConnectTimeout=20', 'otd-h200', cmd], input=stdin, capture_output=True, text=True, timeout=timeout)
    return REDACT.sub('[REDACTED]', r.stdout + r.stderr), r.returncode


SPACE_STATUS = r'''
import json
from pathlib import Path
R = Path('/tmp/mixfull-20261008'); D = Path('/data/workspace/mixfull-20261008'); cfg = json.loads((D / 'launch.json').read_text())
def st(p):
    try: return json.loads(p.read_text())
    except Exception: return {}
s = st(R / 'sweeps' / cfg['tag'] / 'status.json'); done = (R / 'sweeps' / cfg['tag'] / 'train.done')
pid = (D / 'wrapper.pid').read_text().strip() if (D / 'wrapper.pid').exists() else ''
alive = bool(pid) and Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
boot = [l for l in open('/proc/stat') if l.startswith('btime')][0].split()[1]
print('STATUS=' + json.dumps({'boot': boot, 'tmp_ok': R.exists(), 'alive': alive, 'state': s.get('state'), 'stage': s.get('stage'),
      'epoch': s.get('epoch'), 'rows': s.get('logical_rows'), 'total': s.get('total_rows'), 'elapsed': s.get('elapsed_seconds'),
      'loss_ema': s.get('loss_ema'), 'rollbacks': s.get('rollbacks'), 'done': done.read_text().strip() if done.exists() else None}))
'''
H200_STATUS = r'''cd /workspace/woog/pangram/final-20261007; T=moe-mixB-full
B=$(stat -c %Y /proc/1); A=$(cat sweeps/$T.wrapper.pid 2>/dev/null); AL=0; [ -n "$A" ] && kill -0 $A 2>/dev/null && AL=1
Q=$(grep -c "Disk quota\|No space left" sweeps/$T/train.log 2>/dev/null); S=$(cat sweeps/$T/status.json 2>/dev/null || echo '{}')
D=$(cat sweeps/$T/train.done 2>/dev/null)
echo "STATUS={\"boot\": \"$B\", \"alive\": $AL, \"quota_lines\": ${Q:-0}, \"done\": \"$D\", \"status\": $S}"'''


def status_space():
    out = space(SPACE_STATUS); m = re.search(r'STATUS=(\{.*\})', out, re.S)
    return json.loads(m.group(1)) if m else {'error': out[-300:]}


def status_h200():
    out, rc = h200(H200_STATUS); m = re.search(r'STATUS=(\{.*\})', out, re.S)
    if not m: return {'error': out[-300:], 'rc': rc}
    d = json.loads(m.group(1)); s = d.pop('status') or {}
    d.update({k: s.get(k) for k in ('state', 'stage', 'epoch', 'logical_rows', 'total_rows', 'elapsed_seconds', 'loss_ema', 'rollbacks')})
    return d


def free_gb_h200():
    out, _ = h200('du -s --block-size=1G /workspace 2>/dev/null | tail -1', timeout=600)
    try: return round(2000 / 1.0737 - int(out.split()[0]), 1)  # quota is 2000 GB = ~1862 GiB
    except Exception: return None


def wkey(): return netrc.netrc().authenticators('api.wandb.ai')[2]


def relaunch_space():
    out = space('KEY=' + repr(wkey()) + r'''
import os, subprocess, sys
r = subprocess.run([sys.executable, '/data/workspace/mixfull-20261008/relaunch_space.py'], env={**os.environ, 'WANDB_API_KEY': KEY},
                   capture_output=True, text=True, timeout=5400)
print(r.stdout[-600:], r.stderr[-600:])''', timeout=5600)
    say(event='relaunch_space', out=out[-600:])


def relaunch_h200():
    from huggingface_hub import get_token
    out, rc = h200('bash /workspace/woog/pangram/final-20261007/mixgrid/relaunch_h200.sh', stdin=f'{wkey()}\n{get_token()}\n', timeout=1800)
    say(event='relaunch_h200', rc=rc, out=out[-600:])


def finished(s): return s.get('state') in ('trained', 'diverged') and bool(s.get('done'))


start = time.time(); sched = [60, 120, 240, 480, 960, 1920]; prev = {}; boots = {}
while True:
    el = time.time() - start; wait = next((x - el for x in sched if x > el), 3600)
    time.sleep(max(30, wait) if el > 30 else 0)
    for host, get, rel in (('space', status_space, relaunch_space), ('h200', status_h200, relaunch_h200)):
        try:
            s = get()
        except Exception as e:
            say(host=host, error=repr(e)[:300]); continue
        if host == 'h200': s['free_gib'] = free_gb_h200()
        say(host=host, **s)
        rebooted = host in boots and s.get('boot') and s.get('boot') != boots[host]
        if s.get('boot'): boots[host] = s.get('boot')
        stalled = prev.get(host) is not None and s.get('rows' if host == 'space' else 'logical_rows') == prev[host] and s.get('state') == 'training'
        prev[host] = s.get('rows' if host == 'space' else 'logical_rows')
        if finished(s): continue
        if rebooted or (host == 'space' and not s.get('tmp_ok', True)) or (s.get('alive') in (False, 0) and s.get('state') not in (None, 'loading_model')):
            say(host=host, event='relaunching', rebooted=bool(rebooted), alive=s.get('alive'))
            try: rel()
            except Exception as e: say(host=host, event='relaunch_failed', error=repr(e)[:300])
        elif stalled:
            say(host=host, event='stalled', rows=prev[host])
        if host == 'h200' and (s.get('quota_lines') or (s.get('free_gib') is not None and s['free_gib'] < 20)):
            say(host=host, event='QUOTA_WARNING', free_gib=s.get('free_gib'), quota_lines=s.get('quota_lines'))
    if all(finished(x) for x in (status_space(), status_h200())):
        say(event='all_finished'); break
