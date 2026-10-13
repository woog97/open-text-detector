"""Second round: push the v2 split (rows + never-train list, persisted to /data) and the three Fable evaluation sets, then
start score_worker_t21b.py on GPUs 0, 1, 4, 5, 6, 7 (it shares GPUs with first-round scoring if that is still running)."""
import base64, hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent; RS = H.parents[3] / 'research'
DATA = {'t21-heldout-score-rows-v2.jsonl.gz': RS / 't21-heldout-20261006', 't21-heldout-score-rows-v2-clean-for-t21.jsonl.gz': RS / 't21-heldout-20261006',
        't21-heldout-never-train-v2.json': RS / 't21-heldout-20261006', 'polish-eval-v1.jsonl.gz': RS / 'claude-fable-evals-20261006',
        'humanizer-eval-v1.jsonl.gz': RS / 'claude-fable-evals-20261006', 'cowrite-eval-v1.jsonl.gz': RS / 'claude-fable-evals-20261006'}
for name, d in DATA.items():
    b = (d / name).read_bytes(); step = 2_000_000
    run(f"import pathlib; pathlib.Path('/tmp/pangram-t2-20261006/inputs/{name}.part').unlink(missing_ok=True)", timeout=60)
    for i in range(0, len(b), step):
        run(f"open('/tmp/pangram-t2-20261006/inputs/{name}.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + step]).decode()!r}))", timeout=120)
    run(f"""import os, hashlib, shutil
p='/tmp/pangram-t2-20261006/inputs/{name}'; os.replace(p + '.part', p)
assert hashlib.sha256(open(p,'rb').read()).hexdigest() == {hashlib.sha256(b).hexdigest()!r}
shutil.copy(p, '/data/workspace/pangram-t2-20261006/inputs-t21/{name}'); print('ok {name}')""", timeout=60)
start = r'''
import json, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-t2-20261006'); S = R / 'sweeps'
(R / 'score_worker_t21b.py').write_text(SRC)
pids = {g: subprocess.Popen([sys.executable, '-u', str(R / 'score_worker_t21b.py'), str(g)], cwd=R, stdout=open(S / f'score2-gpu{g}.out', 'a'),
                            stderr=subprocess.STDOUT, start_new_session=True).pid for g in (0, 1, 4, 5, 6, 7)}
(S / 't21b-pids.json').write_text(json.dumps(pids)); print(json.dumps(pids))
'''
run('SRC=' + repr((H / 'score_worker_t21b.py').read_text()) + '\n' + start, timeout=60)
