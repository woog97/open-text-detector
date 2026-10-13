"""Freeze and archive the 6,949 validated papers, with resumable immutable receipts."""
import concurrent.futures, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/iclr-round2-upload-20261003'
SOURCE=ROOT/'research/data/openreview_iclr2027_all'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def freeze():
 meta={p['id']:p for p in map(json.loads,(SOURCE/'manifest.jsonl').open())}
 papers=[];files={}
 for chunk in ['000009','000010','000011']:
  folder=SOURCE/'chunks'/chunk
  for row in json.loads((folder/'extraction.json').read_text())['papers']:
   assert not row['verification']['issues']
   pdf=ROOT/row['pdf'];blob=ROOT/row['text_file']
   assert digest(pdf.read_bytes())==row['pdf_sha256']
   raw=blob.read_bytes();assert digest(raw)==row['blob_sha256'];data=decode(raw)
   assert digest(data['text'].encode())==row['text_sha256']
   papers.append({**row,'chunk':chunk,'metadata':meta[row['forum_id']]})
   for path,sha in [(pdf,row['pdf_sha256']),(blob,row['blob_sha256'])]:
    rel=str(path.relative_to(ROOT));files[rel]={'path':rel,'sha256':sha,'bytes':path.stat().st_size}
  for name in ['manifest.json','extraction.json']:
   path=folder/name;raw=path.read_bytes();rel=str(path.relative_to(ROOT));files[rel]={'path':rel,'sha256':digest(raw),'bytes':len(raw)}
 assert len(papers)==6949 and len({p['forum_id'] for p in papers})==6949
 result={'created_at':time.time(),'papers':papers,'files':list(files.values()),'exceptions':[{'id':'JWhMlket3E','reason':'Blank source PDF; excluded from validated 6949 dataset'}]}
 (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False))
 print(json.dumps({'papers':len(papers),'files':len(files),'bytes':sum(f['bytes'] for f in files.values())}),flush=True)
 return result

def upload():
 import upload_iclr_bulk as u
 manifest=json.loads((OUT/'manifest.json').read_text());files=manifest['files']
 path=OUT/'manifest.json';files=files+[{'path':str(path.relative_to(ROOT)),'sha256':digest(path.read_bytes()),'bytes':path.stat().st_size}]
 receipts=OUT/'receipts.jsonl';prior={r['path']:r for r in map(json.loads,receipts.open())} if receipts.exists() else {}
 jobs=[f for f in files if f['path'] not in prior or prior[f['path']]['sha256']!=f['sha256']]
 def one(f):
  p=ROOT/f['path'];assert digest(p.read_bytes())==f['sha256']
  # PDFs stay a single discoverable object even when larger than backup part size.
  if p.suffix=='.pdf':
   raw=p.read_bytes();r=u.put(raw,'papers/'+f['sha256']+'.pdf');return {**f,'parts':[r]}
  r=u.upload((p,f['path']));assert r['sha256']==f['sha256'];return r
 errors=[]
 with receipts.open('a') as log, concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  fs={pool.submit(one,f):f for f in jobs}
  for n,f in enumerate(concurrent.futures.as_completed(fs),1):
   try:r=f.result();prior[r['path']]=r;log.write(json.dumps(r)+'\n');log.flush()
   except Exception as e:errors.append({'path':fs[f]['path'],'error':str(e)[:200]})
   if n%100==0 or n==len(jobs):
    status={'verified':len(prior),'total':len(files),'errors':errors,'updated_at':time.time()};(OUT/'status.json').write_text(json.dumps(status));print(json.dumps({**status,'errors':len(errors)}),flush=True)
 if errors:raise RuntimeError(f'{len(errors)} failed uploads')
 result={'papers':6949,'files':[prior[f['path']] for f in files],'complete':True,'completed_at':time.time()}
 raw=json.dumps(result,separators=(',',':')).encode();key='backups/iclr2027/manifests/'+digest(raw)+'.json';result['remote_manifest']=u.put(raw,key)
 r=u.client().get(u.URL+key);r.raise_for_status();assert r.content==raw
 # Read back deterministic samples of both PDFs and packed extraction records.
 samples=sorted(result['files'],key=lambda f:f['sha256'])[:20]
 for f in samples:
  h=hashlib.sha256()
  for part in f['parts']:
   r=u.client().get(u.URL+part['key']);r.raise_for_status();assert digest(r.content)==part['sha256'];h.update(r.content)
  assert h.hexdigest()==f['sha256']
 result['sample_readbacks']=len(samples)
 (OUT/'completion.json').write_text(json.dumps(result))
 from index_iclr2027_pdfs import main
 main()
 print('COMPLETE: 6949 PDFs and extraction artifacts uploaded and verified',flush=True)
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 freeze() if '--freeze' in sys.argv else upload()
