"""Collect only run metadata; large completed outputs already persist in R2."""
import io
import json
import subprocess
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/classifications/vast-complete-20260925'
s=json.loads((RUN/'run-state.json').read_text())
prefix='research/classifications/vast-complete-20260925/metadata'
r=subprocess.run(s['ssh']+['tar -C /workspace/pangram -czf - '+prefix],capture_output=True,check=False)
assert r.returncode in (0,1), "Metadata retrieval failed"
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
    assert all(m.name==prefix or m.name.startswith(prefix+'/') for m in t.getmembers())
    t.extractall(ROOT,filter='data')
p=RUN/'metadata/status.json'
if p.exists():print(p.read_text())
