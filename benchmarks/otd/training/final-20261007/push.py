"""Push the final-run code to the Space (/data/workspace/final-20261007/code) and optionally start space_setup.py detached.
  python push.py [--setup]"""
import base64, hashlib, io, os, sys, tarfile
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent; T = H.parent
FILES = {**{n: H / n for n in ['runtime.py', 'models.json', 'run_one.sh', 'space_setup.py', 'smoke_tokens.py', 'fit_rows.py']},
         **{n: T / 'overnight-sweep-20261004' / n for n in ['train_sweep.py', 'modeling_sweep.py', 'persist_daemon.py', 'sweep_eval.py', 'cross_model_eval.py']},
         **{n: T / 'backbone-launch-20261003' / n for n in ['data.py', 'adapters_short.py', 'modeling.py']}}


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for n, p in FILES.items(): t.add(p, arcname=n)
    return buf.getvalue()


def push():
    b = bundle(); D = '/data/workspace/final-20261007'
    run(f"import os; os.makedirs('{D}/code', exist_ok=True); p='{D}/code.tgz.part'; os.path.exists(p) and os.remove(p); print('reset')", timeout=60)
    for i in range(0, len(b), 2_000_000):
        run(f"open('{D}/code.tgz.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + 2_000_000]).decode()!r}))", timeout=120)
    run(f"""import os, hashlib, tarfile
p='{D}/code.tgz'; os.replace(p + '.part', p); assert hashlib.sha256(open(p,'rb').read()).hexdigest() == {hashlib.sha256(b).hexdigest()!r}
with tarfile.open(p) as t: t.extractall('{D}/code')
print('pushed', {len(b)}, sorted(os.listdir('{D}/code')))""", timeout=120)


if __name__ == '__main__':
    push()
    if '--setup' in sys.argv:
        run("""import subprocess, sys
p = subprocess.Popen([sys.executable, '-u', '/data/workspace/final-20261007/code/space_setup.py'], cwd='/tmp', stdout=open('/data/workspace/final-20261007/setup.out', 'a'),
                     stderr=subprocess.STDOUT, start_new_session=True); print('setup pid', p.pid)""", timeout=60)
