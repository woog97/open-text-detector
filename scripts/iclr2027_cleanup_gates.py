"""Fail-closed authorization checks for an exact verified PDF manifest."""
import hashlib
from pathlib import Path
def validate_coverage(manifest,ex,hf,cf):
 ids=[p['id'] for p in manifest['papers']]
 assert ids and len(ids)==len(set(ids))
 assert hf['readback_verified'] and hf['private'] and cf['readback_verified'] and cf['website_published'] and ex['verified']
 assert set(ids)==set(hf['ids'])=={p['forum_id'] for p in ex['papers']}=={p['id'] for p in cf['objects']}
 assert len(ex['papers'])==len(ids)==len(cf['objects'])==hf['records']
 refs={p['id']:p for p in cf['objects']}
 for p in manifest['papers']:
  assert refs[p['id']]['pdf_sha256']==p['sha256'] and refs[p['id']]['pdf_bytes']==p['bytes']
 return ids
def authorize_local_path(p,pdf_root):
 path=Path(p['file'])
 assert not path.is_symlink() and path.resolve().is_relative_to(Path(pdf_root).resolve())
 if path.exists():
  raw=path.read_bytes()
  assert len(raw)==p['bytes'] and hashlib.sha256(raw).hexdigest()==p['sha256']
 return path
