"""Verified, resumable ICLR snapshot; preserves active inputs and recovery metadata."""
import concurrent.futures,hashlib,json,os,sqlite3,time
from pathlib import Path
import upload_iclr_bulk as u
ROOT=u.ROOT;OUT=ROOT/os.environ.get('ICLR_SWEEP_DIR','research/iclr-local-sweep-20261003');DATA=ROOT/'research/data/openreview_iclr2027_all'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def discover():
 OUT.mkdir(exist_ok=True);files={};hashes=set();skipped=[]
 def add(p):
  if not p.is_file() or p.is_symlink():return
  rel=str(p.relative_to(ROOT))
  if p.name.endswith(('.lock','.part','-wal','-shm')) or '__pycache__' in p.parts:skipped.append(rel);return
  if p.suffix in ['.py','.pyc','.sh']:return
  st=p.stat();files[rel]={'path':rel,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 for parent in ['research/data','research/cache','research/extractions','research/classifications','research/sources']:
  for base,dirs,names in os.walk(ROOT/parent):
   dirs[:]=[d for d in dirs if d not in ['__pycache__','.venv','node_modules']]
   path=Path(base)
   for name in names:
    if 'iclr' in str((path/name).relative_to(ROOT)).lower():add(path/name)
 for path in (DATA/'chunks').glob('*/manifest.json'):
  for p in json.loads(path.read_text()).get('papers',[]):
   if p.get('sha256'):hashes.add(p['sha256'])
 for path in (DATA/'chunks').glob('*/extraction*.json'):
  d=json.loads(path.read_text())
  if not isinstance(d,dict) or not isinstance(d.get('papers',[]),list):continue
  for p in d.get('papers',[]):
   if isinstance(p,dict) and p.get('pdf_sha256'):hashes.add(p['pdf_sha256'])
 inv=ROOT/'app/.sites-runtime/atlas/pdf-upload-inventory.json'
 if inv.exists():
  for p in json.loads(inv.read_text()):
   if p.get('conference','').lower()=='iclr' or 'iclr' in p.get('source','').lower():hashes.add(p['sha256'])
 for h in hashes:
  folder=ROOT/'research/extractions/positioned/objects'/h
  if folder.exists():
   for p in folder.rglob('*'):add(p)
 # Snapshot databases consistently; retain originals for live ingestion.
 for rel,row in files.items():
  p=ROOT/rel
  if p.suffix in ['.sqlite','.sqlite3','.db']:
   dest=OUT/'sqlite-snapshots'/(hashlib.sha256(rel.encode()).hexdigest()+'.sqlite3');dest.parent.mkdir(exist_ok=True)
   with sqlite3.connect(f'file:{p}?mode=ro',uri=True) as a,sqlite3.connect(dest) as b:a.backup(b)
   row['upload_path']=str(dest.relative_to(ROOT));row['bytes']=dest.stat().st_size
 result={'created_at':time.time(),'files':list(files.values()),'skipped_transient':skipped,'scope':'Local ICLR data, caches and associated positioned extractions; source code and recovery receipts retained'}
 (OUT/'plan.json').write_text(json.dumps(result));print(json.dumps({'files':len(files),'GB':sum(r['bytes'] for r in files.values())/1e9,'transient_preserved':len(skipped)}),flush=True)
def upload():
 plan=json.loads((OUT/'plan.json').read_text());logpath=OUT/'receipts.jsonl';prior={r['path']:r for r in map(json.loads,logpath.open())} if logpath.exists() else {};errors=[]
 def one(row):
  p=ROOT/row.get('upload_path',row['path'])
  if p.suffix.lower()=='.pdf':
   st=p.stat();raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();receipt=u.put(raw,'papers/'+h+'.pdf');assert p.stat().st_mtime_ns==st.st_mtime_ns
   return {'path':row['path'],'bytes':len(raw),'sha256':h,'mtime_ns':st.st_mtime_ns,'parts':[receipt]}
  return u.upload((p,row['path']))
 jobs=[r for r in plan['files'] if r['path'] not in prior]
 with logpath.open('a') as log,concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  fs={pool.submit(one,r):r for r in jobs}
  for n,f in enumerate(concurrent.futures.as_completed(fs),1):
   try:r=f.result();prior[r['path']]=r;log.write(json.dumps(r)+'\n');log.flush()
   except Exception as e:errors.append({'path':fs[f]['path'],'error':str(e)[:180]})
   if n%200==0 or n==len(jobs):
    d={'verified':len(prior),'total':len(plan['files']),'bytes':sum(r['bytes'] for r in prior.values()),'errors':errors,'at':time.time()};(OUT/'status.json').write_text(json.dumps(d));print(json.dumps({**d,'errors':len(errors)}),flush=True)
 result={'files':list(prior.values()),'errors':errors,'created_at':time.time()};raw=json.dumps(result,separators=(',',':')).encode();key='backups/iclr2027/manifests/'+hashlib.sha256(raw).hexdigest()+'.json';result['remote_manifest']=u.put(raw,key)
 r=u.client().get(u.URL+key);r.raise_for_status();assert r.content==raw
 for f in sorted(prior.values(),key=lambda r:r['sha256'])[:20]:
  h=hashlib.sha256()
  for part in f['parts']:
   r=u.client().get(u.URL+part['key']);r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==part['sha256'];h.update(r.content)
  assert h.hexdigest()==f['sha256']
 result['sample_readbacks']=20;(OUT/'completion.json').write_text(json.dumps(result));print('Snapshot and readbacks complete',flush=True)
 from index_iclr2027_pdfs import main
 main()
def cleanup():
 done=json.loads((OUT/('checkpoint.json' if '--checkpoint' in __import__('sys').argv else 'completion.json')).read_text());ready=set();protected=set();pdf_hashes={}
 import fcntl
 active=set();known=set()
 for folder in (DATA/'chunks').iterdir():
  if not folder.is_dir():continue
  for name in ['workflow.lock','extraction-stage.lock','extraction-manager.lock']:
   lock=folder/name
   if not lock.exists():continue
   with lock.open('a') as f:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:active.add(folder.name)
 for f in (DATA/'chunks').glob('*/manifest.json'):
  m=json.loads(f.read_text());ex=f.parent/'extraction.json';ok=ex.exists() and json.loads(ex.read_text()).get('verified',False)
  busy=f.parent.name in active or m.get('source_chunk') in active
  for p in m.get('papers',[]):
   if p.get('file'):
    rel=str(Path(p['file']).relative_to(ROOT));known.add(rel);pdf_hashes[rel]=p.get('sha256')
    if ok:ready.add(rel)
    elif busy:protected.add(rel)
 protected-=ready
 protected_hashes={pdf_hashes[p] for p in protected}
 removed=[];kept=[]
 with (OUT/'deletions.jsonl').open('a') as log:
  for r in done['files']:
   p=ROOT/r['path'];suffix=p.suffix.lower()
   if suffix not in ['.pdf','.pgf','.parquet','.bin','.zip','.tar','.gz'] or r['path'] in protected:kept.append(r['path']);continue
   if str(p).startswith(str(DATA/'download_queue/pdfs')) and r['path'] not in known:kept.append(r['path']);continue
   if not p.exists():continue
   before=p.stat()
   if time.time()-before.st_mtime<120:kept.append(r['path']);continue
   if before.st_size!=r['bytes'] or sha(p)!=r['sha256'] or p.stat().st_mtime_ns!=before.st_mtime_ns:kept.append(r['path']);continue
   if suffix=='.pgf':
    # Keep artifacts for currently unfinished PDFs, even if an older version exists.
    if p.parent.name in protected_hashes:kept.append(r['path']);continue
   assert not p.is_symlink() and all(q['verified'] for q in r['parts'])
   p.unlink();row={'path':r['path'],'sha256':r['sha256'],'bytes':r['bytes'],'allocated_bytes':before.st_blocks*512};removed.append(row);log.write(json.dumps(row)+'\n');log.flush()
 summary={'deleted_files':len(removed),'freed_bytes':sum(r['bytes'] for r in removed),'freed_allocated_bytes':sum(r['allocated_bytes'] for r in removed),'preserved':kept,'at':time.time()};(OUT/'cleanup.json').write_text(json.dumps(summary));print(json.dumps({k:v if k!='preserved' else len(v) for k,v in summary.items()}),flush=True)
def checkpoint():
 rows={r['path']:r for r in map(json.loads,(OUT/'receipts.jsonl').open())}
 result={'files':list(rows.values()),'created_at':time.time(),'partial_snapshot':True}
 raw=json.dumps(result,separators=(',',':')).encode();key='backups/iclr2027/manifests/'+hashlib.sha256(raw).hexdigest()+'.json';result['remote_manifest']=u.put(raw,key)
 r=u.client().get(u.URL+key);r.raise_for_status();assert r.content==raw
 (OUT/'checkpoint.json').write_text(json.dumps(result));print('Recovery checkpoint verified:',len(rows),flush=True)
 cleanup()
if __name__=='__main__':
 import sys
 {'discover':discover,'upload':upload,'cleanup':cleanup,'checkpoint':checkpoint}[sys.argv[1]]()
