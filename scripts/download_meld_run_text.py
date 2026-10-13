# The meld-run Worker was deleted Oct 13, 2026 (as pangram-meld-run); redeploy it from app/wrangler.meld-run.json before use.
"""Download the pinned private dataset using a scoped CDN URL, never an HF account token."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

access=json.loads(Path(sys.argv[1]).read_text());target=Path(sys.argv[2]);temp=target.with_suffix('.partial')
req=urllib.request.Request(access['url'],headers={'User-Agent':'open-text-detector-meld-run/1.0'})
sha=hashlib.sha256();size=0
with urllib.request.urlopen(req,timeout=120) as response,temp.open('wb') as f:
    while chunk:=response.read(8*1024*1024):
        f.write(chunk);sha.update(chunk);size+=len(chunk)
assert size==access['bytes'] and sha.hexdigest()==access['sha256']
temp.replace(target);print('Pinned Hugging Face dataset downloaded and SHA256 verified',flush=True)
