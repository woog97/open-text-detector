"""One command for the final single-pass mix: assemble pools on this Mac (mix_pools.py), upload them with mix_build.py to the
bucket, run mix_build.py on the Space (CPU only, detached) and wait for mix-report.json.

  python mix_run.py            -> /data/workspace/final-20261007/prepared-v2  (+ mix-report.json, mix-build.log)
  python mix_run.py --test     -> /data/workspace/final-20261007/mix-test/prepared-v2 (dry run on whatever exists now)
  extra flags after --         -> passed to mix_build.py (e.g. -- --human-source clean --pool /data/.../pool-early)
Run with: uv run --with requests --with pyarrow --with 'huggingface_hub>=1.0' --with httpx --with websockets python mix_run.py
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run as _run
import contextlib, io


def run(code, timeout=600):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): _run(code, timeout=timeout)
    return buf.getvalue()
H = Path(__file__).resolve().parent; BUCKET = 'open-text-detector/training-storage'; D = '/data/workspace/final-20261007'
argv = sys.argv[1:]; extra = argv[argv.index('--') + 1:] if '--' in argv else []; test = '--test' in argv
out = f'{D}/mix-test' if test else D; local = Path(os.environ.get('MIX_LOCAL', '/tmp/pangram-final-mix-local')); local.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(H / 'mix_pools.py'), str(local)], check=True)
from huggingface_hub import HfApi
files = {'mix-inputs.tgz': (local / 'mix-inputs.tgz').read_bytes(), 'mix_build.py': (H / 'mix_build.py').read_bytes()}
HfApi().batch_bucket_files(BUCKET, add=[(b, f'workspace/final-20261007/mix/{n}') for n, b in files.items()])
shas = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
print(run(f"""import hashlib, json, os, subprocess, sys, time
from pathlib import Path
M = Path('{D}/mix'); shas = {shas!r}
for _ in range(60):
    if all((M / n).exists() and hashlib.sha256((M / n).read_bytes()).hexdigest() == s for n, s in shas.items()): break
    time.sleep(5)
else: raise SystemExit('upload not visible on the mount: ' + str({{n: (M / n).exists() for n in shas}}))
pidf = Path('{out}/mix-build.pid'); Path('{out}').mkdir(parents=True, exist_ok=True)
assert not pidf.exists() or not Path('/proc/' + pidf.read_text().strip()).exists(), 'mix build already running'
p = subprocess.Popen([sys.executable, '-u', str(M / 'mix_build.py'), '--inputs', str(M / 'mix-inputs.tgz'), '--out', '{out}'] + {extra!r},
                     stdout=open('{out}/mix-build.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
pidf.write_text(str(p.pid)); print('mix build pid', p.pid)""", timeout=400))
while True:
    time.sleep(60)
    r = run(f"""from pathlib import Path
L = Path('{out}/mix-build.log'); lines = L.read_text().splitlines() if L.exists() else []
alive = Path('/proc/' + Path('{out}/mix-build.pid').read_text().strip()).exists()
print(alive, '|', lines[-1][:400] if lines else '')
if lines and ('"event": "done"' in lines[-1] or '"event": "failed"' in lines[-1] or not alive):
    print('REPORT', Path('{out}/mix-report.json').read_text() if Path('{out}/mix-report.json').exists() else Path('{out}/mix-build.out').read_text()[-3000:])""", timeout=120)
    print(time.strftime('%H:%M:%S'), r.strip()[:600] if 'REPORT' not in r else r, flush=True)
    if 'REPORT' in r: break
