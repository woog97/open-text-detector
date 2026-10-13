import base64, io, json, os, sys, contextlib
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
OUT = Path(__file__).resolve().parent / 'pulled'
code = r'''
import base64, json
from pathlib import Path
S = Path('/tmp/pangram-splice-20261006/sweeps'); out = {}
for d in sorted(S.glob('q4b-SP*')):
    e = d / 'eval'; r = {}
    for n in ['stage2-epoch2-sentences.npz', 'stage2-epoch2.json']:
        if (e / n).exists(): r[n] = base64.b64encode((e / n).read_bytes()).decode()
    for n in ['wandb-tracking.json', 'history.json', 'status.json', 'stage2-selection.json', 'pruned.json']:
        if (d / n).exists(): r[n] = base64.b64encode((d / n).read_bytes()).decode()
    out[d.name] = r
print('<J>' + json.dumps(out) + '</J>')
'''
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run(code, timeout=300)
s = buf.getvalue(); data = json.loads(s[s.index('<J>') + 3:s.index('</J>')])
for tag, files in data.items():
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (OUT / tag / n).write_bytes(base64.b64decode(b))
    print(tag, sorted(files))
