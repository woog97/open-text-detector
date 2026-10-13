"""Periodic bulk recovery snapshots; independent of request dispatch and filtering."""
import argparse,datetime,hashlib,json,os,tarfile,time
from pathlib import Path
from huggingface_hub import HfApi
import requests
from run_raw_fast import save

def alive(info):
 try:f=Path('/proc',str(info['pid']),'stat').read_text().split();return f[21]==str(info['start_ticks']) and f[2]!='Z'
 except FileNotFoundError:return False

def main(a):
 b=Path(a.base);r=b/'run';out=b/'raw-bulk-checkpoints';out.mkdir(exist_ok=True);api=HfApi(token=os.environ['HF_TOKEN']);seen={};sequence=0
 while True:
  info=json.loads((b/'raw-process.json').read_text());running=alive(info);status=json.loads((r/'raw-production-status.json').read_text()) if (r/'raw-production-status.json').exists() else {}
  finished=not running or status.get('state')=='target_reached'
  files=list((r/'calls').glob('*.json'))+list((r/'raw-documents').glob('*.json'))+list(r.glob('raw-*.json'))
  if finished and (r/'raw-documents.jsonl').exists():files.append(r/'raw-documents.jsonl')
  changed=[p for p in files if seen.get(str(p))!=(p.stat().st_mtime_ns,p.stat().st_size)]
  if changed:
   sequence+=1;name=f'{sequence:05d}-{int(time.time())}.tar.gz';path=out/name;captured={}
   try:
    with tarfile.open(path,'w:gz',compresslevel=1) as tar:
     for p in changed:
      before=(p.stat().st_mtime_ns,p.stat().st_size);tar.add(p,arcname=str(p.relative_to(b)),recursive=False)
      if before==(p.stat().st_mtime_ns,p.stat().st_size):captured[str(p)]=before
    digest=hashlib.sha256(path.read_bytes()).hexdigest();key=a.prefix+'/'+name;api.batch_bucket_files(a.bucket,add=[(str(path),key)])
    h=hashlib.sha256()
    with requests.get('https://huggingface.co/buckets/'+a.bucket+'/resolve/'+key,headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']},stream=True,timeout=(20,180)) as response:
     response.raise_for_status()
     for chunk in response.iter_content(1024*1024):h.update(chunk)
    assert h.hexdigest()==digest
    seen.update(captured);save(b/'raw-checkpoint-status.json',{'state':'verified','key':key,'sha256':digest,'files':len(changed),'saved_documents':status.get('saved_raw_documents'),'final':finished,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
   except Exception as e:
    save(b/'raw-checkpoint-status.json',{'state':'checkpoint_failed_local_files_preserved','error':type(e).__name__+':'+str(e)[:200],'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if finished:return
  if finished:return
  for _ in range(a.interval):
   time.sleep(1)
   if not alive(info):break
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--bucket',required=True);p.add_argument('--prefix',required=True);p.add_argument('--interval',type=int,default=300);main(p.parse_args())
