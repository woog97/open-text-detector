"""Bounded read-only scan of already cached CCCC shards for pinned English SCP tales."""
import argparse,base64,gzip,hashlib,json,time,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlsplit,unquote
REVISION='03a3de5713a0bb23267d26724346508af0f25327'
INDEX_SHA='e8f3ee5970cc6179ddd7820b7bf4d40a10824048ad168063d5474a20cabad7da'
HOSTS={'www.scp-wiki.net','scp-wiki.wikidot.com'}
def slug(url):
 p=urlsplit(url if '://' in url else 'http://'+url)
 if p.hostname not in HOSTS or p.query:return None
 s=unquote(p.path).strip('/')
 return s if s and '/' not in s and ':' not in s else None
class HashingReader:
 def __init__(self,stream):self.stream=stream;self.sha=hashlib.sha256()
 def read(self,n=-1):
  value=self.stream.read(n);self.sha.update(value);return value
 def __getattr__(self,k):return getattr(self.stream,k)
def scan_one(info,base,index):
 source=Path(info['path']);receipt=base/'scan'/ (source.name+'.receipt.json');output=base/'scan'/(source.name+'.tales.jsonl.gz')
 if receipt.exists():
  saved=json.loads(receipt.read_text());assert saved['archive_sha256']==info['sha256'];return saved
 t=time.monotonic();total=0;host_rows=0;matched=0
 with source.open('rb') as original,gzip.open(output.with_suffix('.partial'),'wt',encoding='utf8',compresslevel=1) as out:
  hashed=HashingReader(original)
  with gzip.GzipFile(fileobj=hashed,mode='rb') as stream:
   for line in stream:
    total+=1
    if b'scp-wiki' not in line:continue
    row=json.loads(line);meta=row.get('metadata') or {};key=slug(meta.get('warc_url') or meta.get('url') or '')
    if key is None:continue
    host_rows+=1
    if key not in index:continue
    wrapped={'slug':key,'record':row,'upstream_json_line_base64':base64.b64encode(line).decode(),'source_row':total,'source_file':source.name,'source_dataset':'common-pile/cccc_filtered','source_revision':REVISION,'archive_sha256':info['sha256'],'upstream_json_line_sha256':hashlib.sha256(line).hexdigest()}
    out.write(json.dumps(wrapped,ensure_ascii=False)+'\n');matched+=1
  # Consume any compressed trailer bytes before comparing exact archive identity.
  for chunk in iter(lambda:hashed.read(1024*1024),b''):pass
  assert hashed.sha.hexdigest()==info['sha256'],'archive_hash_mismatch'
 output.with_suffix('.partial').replace(output)
 with output.open('rb') as f:output_sha=hashlib.file_digest(f,'sha256').hexdigest()
 saved={'file':source.name,'archive_sha256':info['sha256'],'archive_bytes':info['bytes'],'rows_scanned':total,'scp_host_rows':host_rows,'known_tale_records':matched,'output':str(output),'output_sha256':output_sha,'elapsed_seconds':time.monotonic()-t}
 receipt.write_text(json.dumps(saved,indent=2));return saved

def main(base,cache,index_path,workers):
 base.mkdir(exist_ok=True);(base/'scan').mkdir(exist_ok=True)
 assert hashlib.sha256(index_path.read_bytes()).hexdigest()==INDEX_SHA;index=json.loads(index_path.read_text());(base/'tales-index.json').write_bytes(index_path.read_bytes())
 with (base/'scan.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (base/'scan-inventory.json').exists():inventory=json.loads((base/'scan-inventory.json').read_text())
  else:
   inventory=[]
   for p in sorted(cache.glob('dolma-cccc-filtered-*.json.gz')):
    mpath=Path(str(p)+'.manifest.json')
    if not mpath.exists():continue
    m=json.loads(mpath.read_text());assert m['repo']=='common-pile/cccc_filtered' and m['revision']==REVISION and p.stat().st_size==m['bytes'];inventory.append({**m,'path':str(p)})
   (base/'scan-inventory.json').write_text(json.dumps(inventory,indent=2))
  start=time.monotonic();results=[]
  with ThreadPoolExecutor(max_workers=workers) as pool:
   futures=[pool.submit(scan_one,info,base,index) for info in inventory]
   for f in as_completed(futures):
    r=f.result();results.append(r);summary={'completed_files':len(results),'target_files':len(inventory),'rows_scanned':sum(x['rows_scanned'] for x in results),'known_tale_records':sum(x['known_tale_records'] for x in results),'elapsed_seconds':time.monotonic()-start,'state':'scanning'};(base/'scan-progress.json').write_text(json.dumps(summary,indent=2));print(json.dumps({**summary,'last_file':r['file'],'last_file_seconds':r['elapsed_seconds']}),flush=True)
  summary['state']='bounded_cached_inventory_scanned';(base/'scan-progress.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args();main(a.base,a.cache,a.index,a.workers)
