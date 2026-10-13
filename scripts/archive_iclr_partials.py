import concurrent.futures,hashlib,json,subprocess,time
import upload_iclr_bulk as u
out=u.ROOT/'research/iclr-local-sweep-20261003'
items=[]
for p in (u.SOURCE/'download_queue').rglob('*.part'):
 if time.time()-p.stat().st_mtime<600:continue
 r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True)
 if r.returncode==1 and not r.stdout:items.append((p,str(p.relative_to(u.ROOT))))
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(u.upload,items))
raw=json.dumps({'files':rows,'kind':'incomplete-response-archives-not-valid-PDFs'},separators=(',',':')).encode();key='backups/iclr2027/manifests/'+hashlib.sha256(raw).hexdigest()+'.json';receipt=u.put(raw,key)
r=u.client().get(u.URL+key);r.raise_for_status();assert r.content==raw
(out/'partial-archive.json').write_text(json.dumps({'files':rows,'remote_manifest':receipt}))
removed=[]
for p,row in zip([p for p,rel in items],rows):
 stat=p.stat();h=hashlib.sha256(p.read_bytes()).hexdigest();r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True)
 if h==row['sha256'] and stat.st_size==row['bytes'] and p.stat().st_mtime_ns==stat.st_mtime_ns and r.returncode==1 and not r.stdout:
  p.unlink();removed.append({'path':row['path'],'bytes':row['bytes'],'sha256':h})
(out/'partial-deletions.json').write_text(json.dumps(removed));print(json.dumps({'archived_files':len(rows),'deleted_files':len(removed),'freed_bytes':sum(r['bytes'] for r in removed)}))
