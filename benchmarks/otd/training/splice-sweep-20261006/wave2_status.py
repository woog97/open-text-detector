import json, glob, os, subprocess, time
print(time.strftime('%H:%M UTC', time.gmtime()))
print(subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout.replace('\n', ' | '))
S = '/tmp/pangram-splice-20261006/sweeps'
for f in sorted(glob.glob(S + '/q4b-*/status.json')):
    t = f.split('/')[-2]
    if not any(k in t for k in ('LLE', 'MIX', 'Arep')): continue
    s = json.load(open(f)); age = int(time.time() - os.path.getmtime(f))
    print(t, s.get('state'), 'st', s.get('stage'), 'ep', s.get('epoch'), 'b', s.get('batch'), '/', s.get('batches'), 'step', s.get('optimizer_steps'), 'ema', s.get('loss_ema'), 'age', age, 'div', os.path.exists(f.replace('status.json', 'divergence.json')))
for g in range(8):
    p = f'{S}/runner-gpu{g}.out'
    if os.path.exists(p):
        tail = open(p, errors='ignore').read().splitlines()[-1:]
        if any(k in ''.join(tail) for k in ('Traceback', 'Error', 'fail')): print('runner', g, tail)
