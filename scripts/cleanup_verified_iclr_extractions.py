import hashlib,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'research/iclr-cloud-backup-20261003';source=root/'research/data/openreview_iclr2027_all'
a=json.loads((out/'extraction-completion.json').read_text());b=json.loads((out/'completion.json').read_text())
assert not a['failures'] and not b['failed']
items=[(root/r['path'],r) for r in a['files']]+[(source/r['path'],r) for r in b['files'] if Path(r['path']).suffix=='.parquet']
freed=0;count=0;skip=[]
with (out/'extraction-deletion-receipts.jsonl').open('w') as log:
 for p,r in items:
  assert p.resolve().is_relative_to(root/'research') and not p.is_symlink() and all(part['verified'] for part in r['parts'])
  if not p.exists():skip.append({'path':str(p),'reason':'absent'});continue
  st=p.stat();data=p.read_bytes();after=p.stat()
  if hashlib.sha256(data).hexdigest()!=r['sha256'] or len(data)!=r['bytes'] or (st.st_ino,st.st_size,st.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):skip.append({'path':str(p),'reason':'changed'});continue
  p.unlink();freed+=st.st_blocks*512;count+=1
  log.write(json.dumps({'path':str(p.relative_to(root)),'sha256':r['sha256'],'allocated_bytes':st.st_blocks*512,'remote_parts':r['parts']})+'\n');log.flush()
summary={'deleted_files':count,'freed_allocated_bytes':freed,'skipped':skip,'finished_at':time.time()};(out/'extraction-deletion-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
