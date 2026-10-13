"""Ship space_units.py + inputs to the training Space and start it as a detached CPU job.

One remote call: files travel base64-encoded inside the code string, the job is started
with nohup in its own process group, and its PID is recorded. Model assets (the spaCy
transformer wheel, Qwen3-Embedding-0.6B) are downloaded on the Space into /data/workspace.
Python deps go into a Space-local venv that reuses the system torch.

Usage: launch_space_units.py PAIRS.jsonl LUNA_SPLITS.jsonl [OUT_NAME EMBED_MODEL GPU_INDEX]
With a GPU index the job runs the embedding model on that one GPU (bf16).
"""
import base64, gzip, os, sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run

HERE = Path(__file__).resolve().parent
ROOT = '/data/workspace/clause-labeling-20261006'
TRF = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl'
files = {'code/space_units.py': HERE / 'space_units.py', 'code/split_spacy.py': HERE / 'split_spacy.py',
         'input/pairs.jsonl': Path(sys.argv[1]), 'input/luna-splits.jsonl': Path(sys.argv[2])}
OUT, MODEL, GPU = (sys.argv[3:6] + [None] * 3)[:3] if len(sys.argv) > 3 else ('out', 'Qwen/Qwen3-Embedding-0.6B', None)
payload = {k: base64.b64encode(gzip.compress(v.read_bytes())).decode() for k, v in files.items()}

job = f"""set -euo pipefail
cd {ROOT}
echo "setup $(date -u +%FT%TZ)"
python -m venv --system-site-packages /tmp/clause-venv2
mkdir -p wheels
/tmp/clause-venv2/bin/pip download -q --no-deps -d wheels '{TRF}'
/tmp/clause-venv2/bin/pip install -q 'spacy==3.8.*' 'spacy-curated-transformers>=0.2.2,<1' wheels/en_core_web_trf-3.8.0-py3-none-any.whl
/tmp/clause-venv2/bin/python -c 'import transformers, torch; print("transformers", transformers.__version__, transformers.__file__, "torch", torch.__version__)'
echo "run $(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES={GPU or ''} HF_HOME=/data/workspace/hf-home /tmp/clause-venv2/bin/python -u code/space_units.py input/pairs.jsonl input/luna-splits.jsonl {OUT} --model-cache /data/workspace/model-cache --threads 32 --embed-model {MODEL} --device {'cuda' if GPU else 'cpu'}
echo "exit $? $(date -u +%FT%TZ)"
"""

code = f"""
import base64, gzip, json, os, subprocess
root = {ROOT!r}
for rel, b in {payload!r}.items():
    p = os.path.join(root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').write(gzip.decompress(base64.b64decode(b)))
open(os.path.join(root, {OUT!r} + '.sh'), 'w').write({job!r})
log = open(os.path.join(root, {OUT!r} + '.log'), 'a')
proc = subprocess.Popen(['bash', os.path.join(root, {OUT!r} + '.sh')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open(os.path.join(root, {OUT!r} + '.pid'), 'w').write(str(proc.pid))
print('started pid', proc.pid, 'files', sorted(os.listdir(os.path.join(root, 'input'))))
"""
run(code, timeout=120)
