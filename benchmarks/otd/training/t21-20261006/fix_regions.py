"""Re-upload the three T2.1 row files with list regions (they were stored as strings), then clear failed claims so workers retry."""
import base64, hashlib, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
RS = Path(__file__).resolve().parents[4] / 'research/t21-heldout-20261006'
for name in ['t21-heldout-score-rows.jsonl.gz', 't21-heldout-score-rows-v2.jsonl.gz', 't21-heldout-score-rows-v2-clean-for-t21.jsonl.gz']:
    b = (RS / name).read_bytes(); step = 2_000_000
    run(f"import pathlib; pathlib.Path('/tmp/pangram-t2-20261006/inputs/{name}.part').unlink(missing_ok=True)", timeout=60)
    for i in range(0, len(b), step):
        run(f"open('/tmp/pangram-t2-20261006/inputs/{name}.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + step]).decode()!r}))", timeout=120)
    run(f"""import os, hashlib, shutil
p='/tmp/pangram-t2-20261006/inputs/{name}'; os.replace(p + '.part', p)
assert hashlib.sha256(open(p,'rb').read()).hexdigest() == {hashlib.sha256(b).hexdigest()!r}
d='/data/workspace/pangram-t2-20261006/inputs-t21/'; shutil.copy(p, d + '{name}'); print('ok {name}')""", timeout=60)
run(r'''
import json, subprocess, sys, time
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'; cleared = []
for f in list(S.glob('q4b-*/score-*.failed')) + list(S.glob('q4b-*/score2-*.failed')):
    f.with_suffix('.claim').unlink(missing_ok=True); f.unlink(); cleared.append(f.parent.name + '/' + f.stem)
r1 = {g: (S / f'score-gpu{g}.log').read_text().strip().splitlines()[-1] for g in (0, 1, 4, 5, 6, 7)}
alive1 = [g for g, l in r1.items() if 'all_claimed' not in l]
new = None
if not alive1:
    new = subprocess.Popen([sys.executable, '-u', str(R / 'score_worker_t21.py'), '0'], cwd=R, stdout=open(S / 'score-t21-gpu0.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True).pid
print(json.dumps({'cleared': cleared, 'round1_workers_not_finished': alive1, 'restarted_round1_worker': new}))
''', timeout=60)
