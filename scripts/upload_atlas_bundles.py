"""Upload Atlas detail objects in 40-file bundles via the atlas-upload worker /bundle endpoint (~10x faster than per-file PUTs).

Run on the Space after build_atlas_details.py; reads OUT/combined-index.parquet and the upload ledger, verifies every
acknowledged size and MD5, then publishes the index to the training bucket. Access via ATLAS_UPLOAD_URL/ATLAS_UPLOAD_TOKEN env vars."""
import hashlib,json,os,struct,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
D=Path('/tmp/pangram-classify-data');OUT=Path('/tmp/pangram-atlas-details-qwen35-4b');ST=D/'after-build-status.json'
URL=os.environ['ATLAS_UPLOAD_URL'];TOKEN=os.environ['ATLAS_UPLOAD_TOKEN']
def status(**k):ST.write_text(json.dumps({**k,'time':time.time()}));print(json.dumps(k),flush=True)
import pyarrow.parquet as pq
idx=pq.read_table(OUT/'combined-index.parquet',columns=['detail_key']).to_pylist()
keys=sorted({r['detail_key'] for r in idx if r['detail_key']})
ledger=OUT/'uploaded.txt';done=set(ledger.read_text().split()) if ledger.exists() else set()
todo=[k for k in keys if k not in done]
bundles=[todo[i:i+40] for i in range(0,len(todo),40)]
status(state='uploading_bundles',objects=len(keys),already=len(keys)-len(todo),bundles=len(bundles))
def send(group):
    bodies=[(OUT/'objects'/k.rsplit('/',1)[1]).read_bytes() for k in group]
    entries=[];off=0
    for k,b in zip(group,bodies):entries.append({'key':k,'offset':off,'length':len(b)});off+=len(b)
    h=json.dumps(entries).encode();payload=struct.pack('>I',len(h))+h+b''.join(bodies)
    for attempt in range(6):
        try:
            req=urllib.request.Request(URL+'/bundle',data=payload,method='POST',headers={'Authorization':'Bearer '+TOKEN,'User-Agent':'open-text-detector-atlas-publisher/1.0'})
            with urllib.request.urlopen(req,timeout=300) as r:acks=json.loads(r.read())
            md5={k:hashlib.md5(b).hexdigest() for k,b in zip(group,bodies)};size={k:len(b) for k,b in zip(group,bodies)}
            if len(acks)!=len(group) or any(a['size']!=size[a['key']] or a['etag'].strip('"')!=md5[a['key']] for a in acks):raise ValueError('ack mismatch')
            return group
        except Exception as e:
            if attempt==5:raise RuntimeError(f'bundle failed: {type(e).__name__}: {str(e)[:120]}') from None
            time.sleep(2**attempt)
started=time.time();n=0;fails=[]
with ThreadPoolExecutor(16) as pool,ledger.open('a') as log:
    futs=[pool.submit(send,g) for g in bundles]
    for f in as_completed(futs):
        try:
            for k in f.result():log.write(k+'\n')
            log.flush();n+=1
        except Exception as e:fails.append(str(e))
        if (n+len(fails))%50==0:print(json.dumps({'bundles_done':n,'failed':len(fails),'seconds':round(time.time()-started)}),flush=True)
final=set(ledger.read_text().split())
missing=[k for k in keys if k not in final]
if missing or fails:status(state='upload_incomplete',missing=len(missing),failures=fails[:5]);raise SystemExit(1)
from huggingface_hub import HfApi
HfApi().batch_bucket_files('open-text-detector/training-storage',add=[(OUT/'combined-index.parquet','workspace/classifications/iclr2027-clean-v2-20261005/qwen35-4b-fast10/atlas-details-index.parquet')])
status(state='complete',objects=len(keys),upload={'bundles':len(bundles),'seconds':round(time.time()-started)},papers=len(idx))
