"""Push selection.json + space_pipeline.py to the Space and start the pipeline detached in /tmp/pangram-pre2023text.
The export token is read from ~/.config/open-text-detector/pdf-export-token and passed only as an environment variable."""
import base64, hashlib, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent
URL = 'https://open-text-detector-pdf-export.woog09.workers.dev'  # not deployed (no such Worker on Oct 13, 2026): deploy app/wrangler.pdf-export.json before use
sel = (H / 'selection.json').read_bytes(); pipe = (H / 'space_pipeline.py').read_text()
run("""import os, shutil
from pathlib import Path
R = Path('/tmp/pangram-pre2023text'); R.mkdir(exist_ok=True)
assert not (R / 'pid').exists() or not Path('/proc/' + (R / 'pid').read_text().strip()).exists(), 'pipeline already running'
for f in R.glob('selection.json.part'): f.unlink()
if not (R / 'code').exists(): shutil.copytree('/tmp/pangram-humancal/code', R / 'code', symlinks=True)
print('prep ok', sorted(os.listdir(R)))""", timeout=300)
for i in range(0, len(sel), 2_000_000):
    run(f"open('/tmp/pangram-pre2023text/selection.json.part','ab').write(__import__('base64').b64decode({base64.b64encode(sel[i:i + 2_000_000]).decode()!r}))", timeout=120)
tok = open(os.path.expanduser('~/.config/open-text-detector/pdf-export-token')).read().strip()
run("PIPE=" + repr(pipe) + "\nTOK=" + repr(tok) + "\nSHA=" + repr(hashlib.sha256(sel).hexdigest()) + "\nURL=" + repr(URL) + """
import os, hashlib, shutil, subprocess, sys
from pathlib import Path
R = Path('/tmp/pangram-pre2023text'); D = Path('/data/workspace/openreview-pre2023-text-20261007'); D.mkdir(parents=True, exist_ok=True)
os.replace(R / 'selection.json.part', R / 'selection.json'); assert hashlib.sha256((R / 'selection.json').read_bytes()).hexdigest() == SHA
(R / 'space_pipeline.py').write_text(PIPE); shutil.copy(R / 'selection.json', D / 'selection.json'); shutil.copy(R / 'space_pipeline.py', D / 'space_pipeline.py')
env = {**os.environ, 'PDF_EXPORT_URL': URL, 'PDF_EXPORT_TOKEN': TOK}
p = subprocess.Popen([sys.executable, '-u', str(R / 'space_pipeline.py')], cwd=R, env=env, stdout=open(R / 'pipeline.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
(R / 'pid').write_text(str(p.pid)); print('started pid', p.pid)""", timeout=120)
