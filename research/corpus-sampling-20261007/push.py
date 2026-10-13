"""Push the sampling code, the builder modules it imports and every held-out list to the Space
(/data/workspace/corpus-sampling-20261007, with repo-relative paths under repo/). Then run argv[1:] there, if given."""
import base64, hashlib, io, os, sys, tarfile
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
ROOT = Path(__file__).resolve().parents[2]; H = Path(__file__).resolve().parent
FILES = ['research/llm-sentence-edits-20261005/build_llm_edits.py', 'research/claude-hosted-edits-20261006/build_hosted.py',
         'research/claude-sections-20261006/build_sections.py', 'research/claude-fullpapers-20261006/build_fullpapers.py',
         'research/llm-sentence-edits-20261005/heldout-eval-never-train.json', 'research/claude-fullpapers-20261006/heldout-never-train.json',
         'research/claude-sections-20261006/heldout-never-train.json', 'research/claude-hosted-edits-20261006/heldout-never-train.json',
         'research/claude-fable-evals-20261006/heldout-never-train.json', 'research/t21-heldout-20261006/t21-heldout-never-train.json',
         'research/t21-heldout-20261006/t21-heldout-never-train-v2.json', 'research/human-calibration-20261006/selection.json',
         'research/openai-writer-handoff-20261006/D-INSTRUCTIONS.md', 'research/openai-writer-handoff-20261006/E-INSTRUCTIONS.md']
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as t:
    for f in FILES: t.add(ROOT / f, arcname='repo/' + f)
    for f in ('build_pool.py', 'sample.py', 'api_writer.py', 'make_items.py', 'wide1_run.py', 'early_claude.py'): t.add(H / f, arcname=f)
b = buf.getvalue(); D = '/data/workspace/corpus-sampling-20261007'
run(f"import os; os.makedirs('{D}', exist_ok=True); p='{D}/push.tgz.part'; os.path.exists(p) and os.remove(p); print('reset')", timeout=60)
for i in range(0, len(b), 2_000_000):
    run(f"open('{D}/push.tgz.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + 2_000_000]).decode()!r}))", timeout=120)
run(f"""import os, hashlib, tarfile
p='{D}/push.tgz'; os.replace(p + '.part', p); assert hashlib.sha256(open(p,'rb').read()).hexdigest() == {hashlib.sha256(b).hexdigest()!r}
with tarfile.open(p) as t: t.extractall('{D}')
print('pushed', {len(b)})""", timeout=120)
if len(sys.argv) > 1:
    cmd = sys.argv[1:]
    run(f"""import subprocess, sys
r = subprocess.run([sys.executable] + {cmd!r}, cwd='{D}', capture_output=True, text=True, timeout=3000)
print(r.stdout[-5000:]); print(r.stderr[-3000:])""", timeout=3100)
