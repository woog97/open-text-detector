"""Restore backed-up inputs, then resume extraction only; never upload or delete."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from upload_iclr_bulk import client,URL
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'research/data/openreview_iclr2027_all'
def main():
 p=argparse.ArgumentParser();p.add_argument('--chunk',required=True);a=p.parse_args();assert a.chunk.isdigit()
 os.chdir(ROOT);chunk=DATA/'chunks'/a.chunk
 lock=(chunk/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=json.loads((chunk/'manifest.json').read_text());ex=json.loads((chunk/'extraction.json').read_text())
 pdfs={x['path']:x for x in map(json.loads,(ROOT/'research/iclr-cloud-backup-20261003/receipts.jsonl').open())}
 blobs={x['path']:x for x in map(json.loads,(ROOT/'research/iclr-cloud-backup-20261003/extraction-receipts.jsonl').open())}
 tasks={}
 for paper in manifest['papers']:
  dest=Path(paper['file']);rel=dest.relative_to(DATA).as_posix();receipt=pdfs[rel];assert receipt['sha256']==paper['sha256'];tasks[dest]=receipt
 for paper in ex['papers']:
  dest=ROOT/paper['text_file'];receipt=blobs[paper['text_file']];assert receipt['sha256']==paper['blob_sha256'];tasks[dest]=receipt
 def restore(item):
  dest,r=item;assert dest.is_relative_to(ROOT)
  if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==r['sha256'];return
  if shutil.disk_usage(DATA).free-r['bytes']<4*1024**3:raise RuntimeError('4 GiB reserve reached during restoration')
  dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.restore-part');h=hashlib.sha256();size=0
  with tmp.open('wb') as f:
   for part in r['parts']:
    for attempt in range(6):
     try:
      response=client().get(URL+part['key']);response.raise_for_status();b=response.content
      assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256'];break
     except Exception:
      if attempt==5:raise
      time.sleep(5*2**attempt)
    f.write(b);h.update(b);size+=len(b)
   f.flush();os.fsync(f.fileno())
  assert size==r['bytes'] and h.hexdigest()==r['sha256'];tmp.replace(dest)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for n,_ in enumerate(pool.map(restore,tasks.items()),1):
   if n%100==0:print(json.dumps({'restored_or_present':n,'total':len(tasks)}),flush=True)
 (chunk/'restore-receipt.json').write_text(json.dumps({'at':time.time(),'files':len(tasks),'source':'research/iclr-cloud-backup-20261003','hashes_verified':True}))
 fcntl.flock(lock,fcntl.LOCK_UN)
 result=subprocess.run([sys.executable,'scripts/run_iclr2027_pipeline.py','--chunk',a.chunk,'--download-extract-only'])
 if result.returncode:subprocess.run([sys.executable,'scripts/repair_iclr2027_chunk.py','--chunk',a.chunk,'--extract-only'],check=True)
if __name__=='__main__':main()
