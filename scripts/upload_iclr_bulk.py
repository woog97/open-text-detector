# The Worker was deleted Oct 13, 2026 (as pangram-iclr-bulk-upload); redeploy it from app/wrangler.iclr-bulk-upload.json before use.
"""Snapshot the local ICLR 2027 collection to R2 through a scoped Worker."""
import concurrent.futures,hashlib,json,os,sqlite3,threading,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/data/openreview_iclr2027_all'
OUT=ROOT/'research/iclr-cloud-backup-20261003'
URL='https://open-text-detector-iclr-bulk-upload.woog09.workers.dev/'
TOKEN=Path('/private/tmp/pangram-iclr-bulk-token').read_text()
local=threading.local();lock=threading.Lock();start=time.time();done_bytes=0;uploaded_bytes=0
CHUNK=64*1024**2

def client():
 if not hasattr(local,'client'):local.client=httpx.Client(timeout=180,headers={'Authorization':'Bearer '+TOKEN},limits=httpx.Limits(max_connections=2,max_keepalive_connections=2))
 return local.client

def put(data,key):
 global uploaded_bytes
 sha=hashlib.sha256(data).hexdigest();md5=hashlib.md5(data).hexdigest()
 for attempt in range(6):
  try:
   r=client().get(URL+key+'?metadata=1')
   if r.status_code==404:r=client().put(URL+key,content=data,headers={'X-Content-SHA256':sha})
   r.raise_for_status();d=r.json()
   assert d['size']==len(data)
   assert d.get('sha256')==sha or d['etag'].strip('"')==md5
   if not d.get('existing'):
    with lock:uploaded_bytes+=len(data)
   return {'key':key,'bytes':len(data),'sha256':sha,'etag':d['etag'],'verified':True}
  except Exception:
   if attempt==5:raise
   time.sleep(min(2**attempt,16))

def upload(item):
 p,rel=item
 for attempt in range(3):
  st=p.stat();blocks=[];h=hashlib.sha256()
  with p.open('rb') as f:
   while True:
    b=f.read(CHUNK)
    if not b:break
    h.update(b);sha=hashlib.sha256(b).hexdigest()
    key=f'papers/{sha}.pdf' if p.suffix=='.pdf' and st.st_size<=CHUNK else f'backups/iclr2027/objects/{sha}'
    blocks.append(put(b,key))
  after=p.stat()
  if (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns):return {'path':rel,'bytes':st.st_size,'sha256':h.hexdigest(),'parts':blocks,'mtime_ns':st.st_mtime_ns}
 raise RuntimeError('Source kept changing')

def main():
 global done_bytes
 OUT.mkdir(exist_ok=True);snapshot=OUT/'sqlite-snapshots';snapshot.mkdir(exist_ok=True)
 items=[];excluded=[]
 for p in sorted(SOURCE.rglob('*')):
  if not p.is_file() or p.is_symlink():continue
  rel=p.relative_to(SOURCE).as_posix()
  if p.name.endswith(('.part','.lock','-wal','-shm')) or '__pycache__' in p.parts:
   excluded.append(rel);continue
  if p.suffix in ('.sqlite3','.sqlite','.db'):
   target=snapshot/(hashlib.sha256(rel.encode()).hexdigest()+'.sqlite3')
   src=sqlite3.connect(f'file:{p}?mode=ro',uri=True);dst=sqlite3.connect(target)
   src.backup(dst);dst.close();src.close();items.append((target,rel))
  else:items.append((p,rel))
 plan={'started_at':start,'source':str(SOURCE),'files':len(items),'bytes':sum(p.stat().st_size for p,r in items),'excluded_transient':excluded,'paths':[r for p,r in items]}
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({k:v for k,v in plan.items() if k not in ('paths','excluded_transient')}),flush=True)
 completed=[json.loads(line) for line in (OUT/'receipts.jsonl').read_text().splitlines()] if (OUT/'receipts.jsonl').exists() else []
 previous={r['path']:r for r in completed};done_bytes=sum(r['bytes'] for r in completed)
 items=[item for item in items if item[1] not in previous];total_files=len(items)+len(completed);failed=[]
 with (OUT/'receipts.jsonl').open('a') as log,concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  futures={pool.submit(upload,item):item[1] for item in items}
  for n,future in enumerate(concurrent.futures.as_completed(futures),1):
   try:
    row=future.result();completed.append(row);done_bytes+=row['bytes'];log.write(json.dumps(row)+'\n');log.flush()
   except Exception as e:failed.append({'path':futures[future],'error':type(e).__name__+': '+str(e)[:200]})
   if n%100==0 or n==len(items):
    status={'completed':len(completed),'total':total_files,'failed':len(failed),'verified_GiB':round(done_bytes/1024**3,3),'new_upload_GiB':round(uploaded_bytes/1024**3,3),'elapsed_seconds':round(time.time()-start),'new_upload_MiB_s':round(uploaded_bytes/1024**2/max(1,time.time()-start),2)}
    (OUT/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 manifest={'format':'iclr2027-local-backup-v1','started_at':start,'finished_at':time.time(),'files':completed,'failed':failed,'excluded_transient':excluded,'note':'Files enumerated at start. SQLite files use consistent backup snapshots. Parts concatenate in order. Local files retained.'}
 raw=json.dumps(manifest,separators=(',',':')).encode();digest=hashlib.sha256(raw).hexdigest();key=f'backups/iclr2027/manifests/{digest}.json'
 record=put(raw,key);read=client().get(URL+key);read.raise_for_status();assert hashlib.sha256(read.content).hexdigest()==digest
 manifest['remote_manifest']=record;(OUT/'completion.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'complete':not failed,'files':len(completed),'failures':failed,'manifest_key':key,'verified_bytes':done_bytes}),flush=True)
 # Refresh conference metadata even when some unrelated backup files failed.
 from index_iclr2027_pdfs import main as refresh_paper_index
 refresh_paper_index()
 if failed:raise SystemExit(1)
if __name__=='__main__':main()
