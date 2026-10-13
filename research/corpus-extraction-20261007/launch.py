"""Push code (git HEAD: backend/open_text_detector_backend + scripts/extract_missing_iclr2027.py), pdf-metadata.json and
corpus_pipeline.py to the Space, then either run a small end-to-end test (`test`) or start the pipeline detached (`run`).
The export token is read from ~/.config/open-text-detector/pdf-export-token and passed only as an environment variable."""
import base64, hashlib, io, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.config/open-text-detector'))
from remote import run
H = Path(__file__).resolve().parent; ROOT = H.parents[1]; mode = sys.argv[1]
URL = 'https://open-text-detector-pdf-export.woog09.workers.dev'  # not deployed (no such Worker on Oct 13, 2026): deploy app/wrangler.pdf-export.json before use
code = subprocess.run(['git', 'archive', 'HEAD', 'backend/open_text_detector_backend', 'scripts/extract_missing_iclr2027.py'], cwd=ROOT, capture_output=True, check=True).stdout
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
files = {'code.tar': code, 'pdf-metadata.json': (H / 'pdf-metadata.json').read_bytes(), 'corpus_pipeline.py': (H / 'corpus_pipeline.py').read_bytes()}
run("""import os
from pathlib import Path
R = Path('/tmp/pangram-corpus'); R.mkdir(exist_ok=True)
assert not (R / 'pid').exists() or not Path('/proc/' + (R / 'pid').read_text().strip()).exists(), 'pipeline already running'
for f in R.glob('*.part'): f.unlink()
print('ok')""", timeout=120)
for name, b in files.items():
    for i in range(0, len(b), 2_000_000):
        run(f"open('/tmp/pangram-corpus/{name}.part','ab').write(__import__('base64').b64decode({base64.b64encode(b[i:i + 2_000_000]).decode()!r}))", timeout=120)
SHAS = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
tok = open(os.path.expanduser('~/.config/open-text-detector/pdf-export-token')).read().strip()
run("SHAS=" + repr(SHAS) + "\nCOMMIT=" + repr(commit) + "\nTOK=" + repr(tok) + "\nURL=" + repr(URL) + "\nMODE=" + repr(mode) + """
import hashlib, os, shutil, subprocess, sys, tarfile, json
from pathlib import Path
R = Path('/tmp/pangram-corpus'); D = Path('/data/workspace/corpus-extraction-20261007')
for n, h in SHAS.items():
    os.replace(R / (n + '.part'), R / n); assert hashlib.sha256((R / n).read_bytes()).hexdigest() == h, n
shutil.rmtree(R / 'code', ignore_errors=True); (R / 'code').mkdir()
with tarfile.open(R / 'code.tar') as t: t.extractall(R / 'code')
(R / 'code/COMMIT').write_text(COMMIT)
shutil.copy(R / 'pdf-metadata.json', D / 'pdf-metadata.json'); shutil.copy(R / 'corpus_pipeline.py', D / 'corpus_pipeline.py')
env = {**os.environ, 'PDF_EXPORT_URL': URL, 'PDF_EXPORT_TOKEN': TOK}
if MODE == 'run':
    p = subprocess.Popen([sys.executable, '-u', str(R / 'corpus_pipeline.py')], cwd=R, env=env, stdout=open(R / 'pipeline.out', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    (R / 'pid').write_text(str(p.pid)); print('started pid', p.pid)
else:
    test = '''
import sys, json, random, time
sys.path.insert(0, "/tmp/pangram-corpus"); import corpus_pipeline as cp
cp.R = cp.Path("/tmp/pangram-corpus-test"); cp.PDFS = cp.R / "pdfs"; cp.R.mkdir(exist_ok=True); cp.log = open(cp.R / "pipeline.log", "a")
sys.path[:0] = [str(cp.CODE / "backend"), cp.VENDOR]
inv = json.loads((cp.D / "inventory.json").read_text()); r2 = inv["r2"]
rows = cp.canonical(r2); by = {}
for s, r in rows.items(): by.setdefault(r["dataset"], []).append(s)
print("canonical", len(rows), {k: len(v) for k, v in by.items()})
pick = [s for v in by.values() for s in random.Random(1).sample(sorted(v), min(5, len(v)))]
def local(s, r):
    c = [cp.Path(r["root"]) / r["pdf_path"]] if r.get("pdf_path") else []
    return next((x for x in c + [cp.PDFS / f"{s}.pdf"] if x.exists()), None)
bad = cp.fetch_all([s for s in pick if local(s, rows[s]) is None]); print("fetch failed", len(bad))
t0 = time.time(); objs = cp.R / "objs"
for s in pick:
    p = local(s, rows[s]); sha, d, cl, om, err = cp.process((s, rows[s], str(p or ""), str(objs)))
    print(rows[s]["dataset"][:30], s[:10], "err", err, "| direct", len(d["text"]) if d else None, "clean", len(cl["text"]) if cl else None,
          "omit", len(om["text"]) if om else None, om["omission_counts_json"] if om else "")
print("seconds", round(time.time() - t0))
'''
    r = subprocess.run([sys.executable, '-c', test], env=env, capture_output=True, text=True, timeout=1500)
    print(r.stdout[-6000:]); print(r.stderr[-3000:])
""", timeout=1800)
