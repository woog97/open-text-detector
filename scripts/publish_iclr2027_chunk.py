"""Publish and verify the frozen first ICLR chunk; never infer classifications."""
import sys,json,hashlib,gzip,time,urllib.request,tomllib,fcntl,threading,concurrent.futures,os
from pathlib import Path
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'scripts')]
from open_text_detector_backend.result_codec import decode
from process_new_positioned_papers import verify_one,write
from upload_paper_pdfs import BASE,CONFIG,refresh
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi,hf_hub_download
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--chunk',required=True);args=parser.parse_args();assert args.chunk.isdigit()
root=ROOT/'research/data/openreview_iclr2027_all';chunk=root/'chunks'/args.chunk
lock=(chunk/'processing.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(b):return hashlib.sha256(b).hexdigest()
token_lock=threading.Lock()
def r2(key,data=None,ctype='application/octet-stream'):
 for attempt in range(5):
  with token_lock:
   refresh()
   token=tomllib.loads(CONFIG.read_text())['oauth_token']
  req=urllib.request.Request(BASE+'/'+key,data=data,method='PUT' if data is not None else 'GET',headers={'Authorization':'Bearer '+token,'Content-Type':ctype})
  try:
   with urllib.request.urlopen(req,timeout=120) as r:return r.read()
  except urllib.error.HTTPError as e:
   if e.code==401 and attempt==0:
    with token_lock:refresh(force=True)
    continue
   if e.code==429:
    delay=e.headers.get('Retry-After','60');time.sleep(int(delay) if delay.isdigit() else 60);continue
   if e.code in (500,502,503,504) and attempt<4:time.sleep(2**attempt);continue
   raise RuntimeError('R2 HTTP '+str(e.code)) from None
 raise RuntimeError('R2 retries exhausted')
manifest=json.loads((chunk/'manifest.json').read_text());ex=json.loads((chunk/'extraction.json').read_text());assert ex['verified']
assert {p['id'] for p in manifest['papers']}=={p['forum_id'] for p in ex['papers']} and len(manifest['papers'])==len(ex['papers'])
meta={p['id']:p for p in map(json.loads,(root/'manifest.jsonl').open())}
rows=[];objects=[]
for p in ex['papers']:
 assert not verify_one(p)['issues']
 blob=(ROOT/p['text_file']).read_bytes();d=decode(blob);m=meta[p['forum_id']]
 rows.append(dict(paper_id=p['pdf_sha256'],forum_id=p['forum_id'],title=m['title'],conference='iclr',year=2027,text=d['text'],pdf_sha256=p['pdf_sha256'],text_sha256=p['text_sha256'],extraction_sha256=p['blob_sha256'],extraction_method=d['method'],extraction_metadata_json=json.dumps({k:d[k] for k in ['mapping','diagnostics','warnings','ocr_pages','preprocessing','geometry_version'] if k in d}),page_count=len(d['pages'])))
 objects.append(p)
api=HfApi();repo='open-text-detector/paper-text';info=api.repo_info(repo,repo_type='dataset');assert info.private
schema=pa.schema([('paper_id',pa.string()),('forum_id',pa.string()),('title',pa.string()),('conference',pa.string()),('year',pa.int32()),('text',pa.large_string()),('pdf_sha256',pa.string()),('text_sha256',pa.string()),('extraction_sha256',pa.string()),('extraction_method',pa.string()),('extraction_metadata_json',pa.string()),('page_count',pa.int32())])
shard=chunk/'canonical.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=schema),shard,compression='zstd')
h=sha(shard.read_bytes());key='data/iclr2027-'+h+'.parquet';files=api.list_repo_files(repo,repo_type='dataset',revision=info.sha)
if key not in files:
 commit=api.upload_file(path_or_fileobj=str(shard),path_in_repo=key,repo_id=repo,repo_type='dataset',parent_commit=info.sha,commit_message='Add verified ICLR 2027 canonical text shard').oid
else:commit=info.sha
assert api.repo_info(repo,repo_type='dataset',revision=commit).private
remote=Path(hf_hub_download(repo_id=repo,repo_type='dataset',filename=key,revision=commit,force_download=True))
assert sha(remote.read_bytes())==h
remote_rows=pq.read_table(remote).to_pylist();assert remote_rows==rows
assert set(files)<=set(api.list_repo_files(repo,repo_type='dataset',revision=commit))
hf={'repo':repo,'private':True,'commit':commit,'shard':key,'sha256':h,'records':len(rows),'ids':[r['forum_id'] for r in rows],'readback_verified':True}
write(chunk/'hf-receipt.json',hf);print('Private HF shard verified',flush=True)
# Make submission metadata discoverable before individual PDFs finish publishing.
from index_iclr2027_pdfs import main as index_uploaded_pdfs
index_uploaded_pdfs()
old_raw=r2('atlas-public/catalogue.json');catalogue=json.loads(old_raw);write(chunk/'site-catalogue-before.json',catalogue)
refs=[]
object_receipts=chunk/'r2-objects';object_receipts.mkdir(exist_ok=True)
def publish_object(p):
 saved=object_receipts/(p['forum_id']+'.json')
 if saved.exists():
  cached=json.loads(saved.read_text());assert cached.get('readback_verified');ref=cached['ref'];item=cached['item']
  assert ref['pdf_sha256']==p['pdf_sha256'] and ref['text_sha256']==p['text_sha256']
  return ref,item
 d=decode((ROOT/p['text_file']).read_bytes());m=meta[p['forum_id']];blob=(ROOT/p['text_file']).read_bytes()
 pdf=(ROOT/p['pdf']).read_bytes();assert sha(pdf)==p['pdf_sha256']
 pdfkey='papers/'+p['pdf_sha256']+'.pdf'
 detail={'id':p['pdf_sha256'][:24],'version':p['pdf_sha256'],'version_verified':True,'forum_id':p['forum_id'],'title':m['title'],'text':d['text'],'text_sha256':p['text_sha256'],'reports':[],'position_maps':{p['text_sha256']:{'pages':d['pages'],'rectangles':d['rectangles']}},'extraction':{'method':d['method'],'mapping':d['mapping'],'geometry_version':d['geometry_version']}}
 compressed=gzip.compress(json.dumps(detail,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)
 detailkey='atlas-public/objects/'+sha(compressed)+'.json.gz';mapkey='extractions/positioned/'+p['pdf_sha256']+'/'+p['text_sha256']+'.pgf'
 for k,data,ctype in [(pdfkey,pdf,'application/pdf'),(mapkey,blob,'application/octet-stream'),(detailkey,compressed,'application/gzip')]:
  r2(k,data,ctype);read=r2(k);assert len(read)==len(data) and sha(read)==sha(data)
  if k==mapkey:assert decode(read)==d
  if k==detailkey:assert json.loads(gzip.decompress(read))==detail
 item={'id':detail['id'],'title':m['title'],'filename':p['forum_id']+'.pdf','collection':'iclr/2027','bytes':len(pdf),'classified':False,'models':[],'score_summaries':{},'pdf_key':pdfkey,'detail_key':detailkey}
 ref={'id':p['forum_id'],'site_id':item['id'],'pdf_key':pdfkey,'pdf_sha256':sha(pdf),'pdf_bytes':len(pdf),'detail_key':detailkey,'detail_sha256':sha(compressed),'detail_bytes':len(compressed),'position_key':mapkey,'position_sha256':sha(blob),'position_bytes':len(blob),'text_sha256':p['text_sha256']}
 write(saved,{'ref':ref,'item':item,'readback_verified':True})
 return ref,item
# Bound concurrency and memory; catalogue mutation remains serialized in this thread.
workers=max(1,min(12,int(os.environ.get('ICLR_UPLOAD_WORKERS','8'))))
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
 for ref,item in pool.map(publish_object,objects):
  existing=[v for v in catalogue['items'] if v['id']==item['id']]
  if existing:assert existing==[item]
  else:catalogue['items'].append(item)
  refs.append(ref)
  if len(refs)%25==0:print(json.dumps({'r2_verified':len(refs),'total':len(objects),'workers':workers}),flush=True)
# Recheck the live index to avoid replacing an intervening publisher's changes.
assert r2('atlas-public/catalogue.json')==old_raw,'Site catalogue changed; rerun merge'
catalogue['published_at']=time.time()
for model in catalogue.get('models',[]):
 model['total']=len(catalogue['items']);model['complete']=model.get('available')==model['total']
raw=json.dumps(catalogue,separators=(',',':')).encode();r2('atlas-public/catalogue.json',raw,'application/json');assert r2('atlas-public/catalogue.json')==raw
cf={'objects':refs,'index_key':'atlas-public/catalogue.json','index_sha256':sha(raw),'prior_index_sha256':sha(old_raw),'records':len(refs),'website_published':True,'readback_verified':True,'worker_verified':False}
write(chunk/'cloudflare-receipt.json',cf)
print('HF and R2 readback verified; deployed worker verification is next',flush=True)
