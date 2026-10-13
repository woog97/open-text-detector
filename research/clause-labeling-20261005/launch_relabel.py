"""Ship space_relabel.py + a compact row file to the Space and start it as a detached job.

Usage: launch_relabel.py ROWS.jsonl OUT_NAME [GPU_INDEX]
       launch_relabel.py remote:input/NAME.jsonl.gz OUT_NAME [GPU_INDEX]   (rows already in the bucket under ROOT)
Same setup as launch_space_units.py (Space-local venv reusing system torch; model assets in /data/workspace).
"""
import base64, gzip, os, sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run

HERE = Path(__file__).resolve().parent
ROOT = '/data/workspace/clause-labeling-20261006'
TRF = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl'
rows, OUT = sys.argv[1], sys.argv[2]
rows = rows if rows.startswith('remote:') else Path(rows)
GPU = sys.argv[3] if len(sys.argv) > 3 else None
files = {f'code/{n}': HERE / n for n in ('space_relabel.py', 'space_units.py', 'split_spacy.py')}
REMOTE = str(rows).startswith('remote:')
if not REMOTE:
    files[f'input/{OUT}.jsonl'] = rows
unpack = f"gunzip -c {str(rows)[7:]} > input/{OUT}.jsonl" if REMOTE else 'true'
payload = {k: base64.b64encode(gzip.compress(v.read_bytes())).decode() for k, v in files.items()}
job = f"""set -euo pipefail
cd {ROOT}
echo "setup $(date -u +%FT%TZ)"
python -m venv --system-site-packages /tmp/clause-venv2
mkdir -p wheels
[ -f wheels/en_core_web_trf-3.8.0-py3-none-any.whl ] || /tmp/clause-venv2/bin/pip download -q --no-deps -d wheels '{TRF}'
/tmp/clause-venv2/bin/pip install -q 'spacy==3.8.*' 'spacy-curated-transformers>=0.2.2,<1' wheels/en_core_web_trf-3.8.0-py3-none-any.whl
{unpack}
echo "run $(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES={GPU or ''} HF_HOME=/data/workspace/hf-home /tmp/clause-venv2/bin/python -u code/space_relabel.py input/{OUT}.jsonl {OUT} --model-cache /data/workspace/model-cache --threads {os.environ.get('RELABEL_THREADS', '32')} --device {'cuda' if GPU else 'cpu'}
echo "exit $? $(date -u +%FT%TZ)"
"""
code = f"""
import base64, gzip, os, subprocess
root = {ROOT!r}
for rel, b in {payload!r}.items():
    p = os.path.join(root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').write(gzip.decompress(base64.b64decode(b)))
open(os.path.join(root, {OUT!r} + '.sh'), 'w').write({job!r})
log = open(os.path.join(root, {OUT!r} + '.log'), 'a')
proc = subprocess.Popen(['bash', os.path.join(root, {OUT!r} + '.sh')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open(os.path.join(root, {OUT!r} + '.pid'), 'w').write(str(proc.pid))
print('started pid', proc.pid)
"""
run(code, timeout=120)
