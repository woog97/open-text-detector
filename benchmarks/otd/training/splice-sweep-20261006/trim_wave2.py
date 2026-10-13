import json, os, signal, time
from pathlib import Path
S = Path('/tmp/pangram-splice-20261006/sweeps')
pids = json.loads((S / 'runner-pids-wave2.json').read_text())
drop = [f'q4b-{a}-s{s}' for a in ('LLE', 'MIX', 'Arep') for s in (2, 3)]
q = json.loads((S / 'queue.json').read_text())
(S / 'queue.json').write_text(json.dumps([j for j in q if j['tag'] not in drop], indent=1))
killed = {}
for t in drop:
    c = S / t / 'claim.json'
    if c.exists():
        g = str(json.loads(c.read_text())['gpu']); rp = pids['runners'][g]
        try: os.killpg(rp, signal.SIGTERM); killed[t] = (g, rp)
        except ProcessLookupError: killed[t] = (g, 'gone')
        (S / t / 'cancelled.json').write_text(json.dumps({'reason': 'user: one seed per arm', 't': time.time()}))
time.sleep(5)
alive = {g: os.path.exists(f'/proc/{p}') for g, p in pids['runners'].items()}
print(json.dumps({'killed': killed, 'runner_alive': alive, 'queue_left': [j['tag'] for j in json.loads((S / 'queue.json').read_text()) if not (S / j['tag'] / 'claim.json').exists()]}))
