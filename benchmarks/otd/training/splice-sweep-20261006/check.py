import json, os, subprocess, time
from pathlib import Path
S = Path('/tmp/pangram-splice-20261006/sweeps'); pids = json.loads((S / 'runner-pids.json').read_text())
print('runners alive:', {g: os.path.exists(f'/proc/{p}') for g, p in pids.items()})
for d in sorted(S.glob('q4b-*')):
    st = json.loads((d / 'status.json').read_text()) if (d / 'status.json').exists() else {}
    age = int(time.time() - (d / 'status.json').stat().st_mtime) if (d / 'status.json').exists() else None
    tl = (d / 'train.log').read_text(errors='ignore') if (d / 'train.log').exists() else ''
    errs = sum(tl.count(k) for k in ['Traceback', 'Error', 'out of memory'])
    ev = sorted(p.stem for p in (d / 'eval').glob('*.json')) if (d / 'eval').exists() else []
    print(d.name, {k: st.get(k) for k in ['state', 'stage', 'epoch', 'batch', 'batches', 'optimizer_steps', 'loss_ema', 'loss_ema_min', 'grad_norm']}, 'age', age, 'errs', errs,
          'diverged' if (d / 'divergence.json').exists() else '', 'eval', ev)
    if errs: print(tl[-1500:])
    st_ = st.get('state')
    if errs or (d / 'divergence.json').exists() or st_ in ('failed', 'diverged') or (st_ in ('training', 'loading_model') and age and age > 900):
        print('PROBLEM', d.name)
for g in pids: print(g, (S / f'runner-gpu{g}.log').read_text().strip().splitlines()[-1:] if (S / f'runner-gpu{g}.log').exists() else '')
print(subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout)
print(time.strftime('%H:%M UTC', time.gmtime()))
