"""Remove only unchanged, checksum-verified ICLR bulk files backed up this session."""
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/iclr-cloud-backup-20261003';SOURCE=ROOT/'research/data/openreview_iclr2027_all'
main=json.loads((OUT/'completion.json').read_text());assert not main['failed']
items=[(SOURCE/r['path'],r) for r in main['files'] if Path(r['path']).suffix in ('.pdf','.bin')]
items.sort(key=lambda item:item[1]['bytes'],reverse=True)
freed=0;removed=0;skipped=[]
def remove(item):
 p,r=item
 if not p.exists():return {'path':str(p.relative_to(ROOT)),'status':'already_absent','bytes':0}
 assert not p.is_symlink() and p.resolve().is_relative_to(SOURCE.resolve())
 assert sum(part['bytes'] for part in r['parts'])==r['bytes'] and all(part['verified'] for part in r['parts'])
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 after=p.stat()
 if before.st_size!=r['bytes'] or h.hexdigest()!=r['sha256'] or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):return {'path':str(p.relative_to(ROOT)),'status':'changed_preserved','bytes':0}
 p.unlink();return {'path':str(p.relative_to(ROOT)),'status':'deleted','bytes':r['bytes'],'allocated_bytes':before.st_blocks*512,'sha256':r['sha256'],'remote_parts':r['parts']}
with (OUT/'local-deletion-receipts.jsonl').open('a') as log,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for n,r in enumerate(pool.map(remove,items),1):
  log.write(json.dumps(r)+'\n');log.flush()
  if r['status']=='deleted':freed+=r['allocated_bytes'];removed+=1
  else:skipped.append(r)
  if n%250==0 or n==len(items):print(json.dumps({'deleted_files':removed,'freed_GiB':round(freed/1024**3,2),'processed':n,'total':len(items)}),flush=True)
summary={'deleted_files':removed,'freed_allocated_bytes':freed,'skipped':skipped,'retained':'Tracking databases, manifests, extraction files and other non-PDF/non-response artifacts remain.','finished_at':time.time()}
(OUT/'local-deletion-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
