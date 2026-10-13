import json, os, subprocess, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'
print(time.strftime('%H:%M UTC', time.gmtime()), 'uptime_s', open('/proc/uptime').read().split()[0])
p = json.loads((S / 't21-pids.json').read_text()); print('run_t21 alive', os.path.exists(f"/proc/{p['run_t21_pid']}"))
if (S / 't21-build.failed').exists(): print('PROBLEM build failed'); print((R / 'build-t21.out').read_text()[-1500:])
bl = (R / 'build-t21.log').read_text().strip().splitlines() if (R / 'build-t21.log').exists() else []
for l in bl[-3:]: print('build', l[:600])
for d in sorted(S.glob('q4b-T21-*')):
    st = json.loads((d / 'status.json').read_text()) if (d / 'status.json').exists() else {}
    age = int(time.time() - (d / 'status.json').stat().st_mtime) if (d / 'status.json').exists() else None
    tl = (d / 'train.log').read_text(errors='ignore') if (d / 'train.log').exists() else ''
    errs = sum(tl.count(k) for k in ['Traceback', 'out of memory'])
    print(d.name, {k: st.get(k) for k in ['state', 'stage', 'epoch', 'batch', 'batches', 'optimizer_steps', 'loss_ema', 'loss_ema_min']}, 'age', age, 'errs', errs,
          'done' if (d / 'train.done').exists() else '', 'diverged' if (d / 'divergence.json').exists() else '',
          sorted(x.name for x in (d / 'eval').glob('*.json')) if (d / 'eval').exists() else [])
    if errs: print(tl[-1500:])
    if errs or (d / 'divergence.json').exists() or (st.get('state') == 'training' and age and age > 900): print('PROBLEM', d.name)
for f in list(S.glob('q4b-*/score-*.failed')) + list(S.glob('q4b-*/score2-*.failed')): print('PROBLEM', f)
r2 = [(t, n) for t in [f'q4b-T21-s{s}' for s in (1, 2, 3)] + [f'q4b-T2-s{s}' for s in (1, 2, 3)] + [f'q4b-SPG-s{s}' for s in (1, 2, 3)]
      for n in ('t21-heldout-v2-clean' if 'T21' in t else 't21-heldout-v2', 'fable-polish', 'fable-humanizer', 'fable-cowrite')]
print('ROUND2 done', sum((S / t / 'eval' / f'{n}-stage2-epoch2.json').exists() for t, n in r2), 'of', len(r2))
if (S / 't21b-pids.json').exists(): print('round2 workers alive', sum(os.path.exists(f'/proc/{v}') for v in json.loads((S / 't21b-pids.json').read_text()).values()))
for g in (0, 1, 4, 5, 6, 7):
    for pre in ('score', 'score2'):
        f = S / f'{pre}-gpu{g}.log'; ls = f.read_text().strip().splitlines() if f.exists() else []
        if ls: print(pre, g, ls[-1][:160])
print(subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout)
