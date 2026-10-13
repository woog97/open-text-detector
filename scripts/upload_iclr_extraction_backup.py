"""Back up local extraction files referenced by ICLR 2027 chunk manifests."""
import concurrent.futures,hashlib,json,time
import upload_iclr_bulk as u

def main():
 paths=set()
 for f in (u.SOURCE/'chunks').glob('*/extraction.json'):
  for row in json.loads(f.read_text()).get('papers',[]):
   if row.get('text_file'):paths.add(row['text_file'])
 items=[];missing=[]
 for rel in sorted(paths):
  p=u.ROOT/rel
  assert p.resolve().is_relative_to((u.ROOT/'research/extractions').resolve())
  if p.is_file():items.append((p,rel))
  else:missing.append(rel)
 print(json.dumps({'extraction_files':len(items),'bytes':sum(p.stat().st_size for p,r in items),'missing_references':len(missing)}),flush=True)
 rows=[];failures=[]
 with (u.OUT/'extraction-receipts.jsonl').open('w') as log,concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  fs={pool.submit(u.upload,item):item[1] for item in items}
  for n,f in enumerate(concurrent.futures.as_completed(fs),1):
   try:r=f.result();rows.append(r);log.write(json.dumps(r)+'\n');log.flush()
   except Exception as e:failures.append({'path':fs[f],'error':type(e).__name__})
   if n%500==0:print(json.dumps({'verified':len(rows),'total':len(items),'failures':len(failures)}),flush=True)
 manifest={'format':'iclr2027-associated-extractions-v1','files':rows,'missing_references':missing,'failures':failures}
 raw=json.dumps(manifest,separators=(',',':')).encode();key='backups/iclr2027/manifests/'+hashlib.sha256(raw).hexdigest()+'.json';receipt=u.put(raw,key)
 r=u.client().get(u.URL+key);r.raise_for_status();assert r.content==raw
 manifest['remote_manifest']=receipt;(u.OUT/'extraction-completion.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'verified':len(rows),'bytes':sum(r['bytes'] for r in rows),'failures':failures,'missing_references':len(missing),'manifest_key':key}),flush=True)
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
