#!/bin/bash
# Usage: health_run.sh TAG — checks an H200 sweep run every 5 min; exits (waking the session) on any problem or when the run ends.
# Problems: stale status, errors in log, divergence (trainer guard or loss EMA > 1.5x its minimum + 0.05), quota, /dev/shm > 92%.
TAG=$1
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 otd-h200 "
    B=/workspace/woog/pangram/backbones-20261003; R=\$B/sweeps/$TAG
    python3 - \$R <<'EOF'
import json, os, sys, time, subprocess
R = sys.argv[1]; s = json.load(open(R + '/status.json'))
age = int(time.time() - os.path.getmtime(R + '/status.json'))
log = open(R + '/train.log', errors='ignore').read() if os.path.exists(R + '/train.log') else ''
errs = sum(log.count(k) for k in ['Traceback', 'Disk quota', 'out of memory'])
shm = int(subprocess.run(['df', '--output=pcent', '/dev/shm'], capture_output=True, text=True).stdout.split()[-1].rstrip('%'))
try:
    with open(R + '/../.qtest', 'wb') as f: f.write(b'0' * (256 << 20))
    q = 'ok'
except OSError: q = 'FAIL'
finally:
    try: os.remove(R + '/../.qtest')
    except OSError: pass
ema, mn = s.get('loss_ema'), s.get('loss_ema_min')
warn = 'yes' if (ema and mn and ema > 1.5 * mn + .05) else 'no'
print(f\"age={age} state={s.get('state')} step={s.get('optimizer_steps')} stage={s.get('stage')} epoch={s.get('epoch')} batch={s.get('batch')}/{s.get('batches')} \"
      f\"loss_ema={ema} min={mn} gnorm={s.get('grad_norm')} lr={s.get('learning_rate')} errs={errs} shm={shm} quota={q} drift={warn} \"
      f\"done={'yes' if os.path.exists(R + '/wrapper.done') else 'no'} diverged={'yes' if os.path.exists(R + '/divergence.json') else 'no'}\")
EOF" 2>&1)
  echo "$(TZ=America/Los_Angeles date +%H:%M) $out"
  case "$out" in
    *done=yes*|*quota=FAIL*|*diverged=yes*|*drift=yes*|*state=failed*|*state=diverged*) exit 0;;
  esac
  age=$(echo "$out" | sed -n 's/.*age=\([0-9]*\).*/\1/p'); errs=$(echo "$out" | sed -n 's/.*errs=\([0-9]*\).*/\1/p'); shm=$(echo "$out" | sed -n 's/.*shm=\([0-9]*\).*/\1/p')
  if [ -z "$age" ] || [ "$age" -gt 600 ] || [ "${errs:-0}" -gt 0 ] || [ "${shm:-0}" -gt 92 ]; then exit 0; fi
  sleep 300
done
