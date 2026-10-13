"""Counterbalanced, isolated real-source ingestion benchmark; never training intake."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,json,subprocess,sys,time,statistics
from collect_pool import atomic_json,now
from expand_pool import connect,counts

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--quota',type=int,default=1000);p.add_argument('--run-name',default='parallel-originals-warm-v2')
 a=p.parse_args();b=a.base;root=b/'performance'/a.run_name;root.mkdir(parents=True,exist_ok=False)
 plan=json.loads((b/'pipeline/sampling-plan.json').read_text());records=[]
 from ingest_original_archives import writingprompts_documents
 warm_start=time.perf_counter();d=b/'source-downloads/writingprompts';g=writingprompts_documents(d/'original.tar.gz',d/'train-cache');next(g);g.close();warmup=time.perf_counter()-warm_start
 # A-B-B-A minimizes monotonic machine load/cache drift. Source files are already downloaded.
 for trial,workers in enumerate([1,2,2,1]):
  dest=root/('trial-'+str(trial));(dest/'pipeline').mkdir(parents=True);(dest/'progress').mkdir()
  (dest/'source-downloads').symlink_to(b/'source-downloads',target_is_directory=True)
  subplan={**plan,'source_quotas':[{**s,'planned_passages':a.quota} for s in plan['source_quotas'] if s['source_id'] in ['imdb','writingprompts']]}
  atomic_json(dest/'pipeline/sampling-plan.json',subplan);db=connect(dest/'collection.sqlite3');db.close()
  def run(sid):
   started=time.perf_counter()
   with (dest/(sid+'.log')).open('w') as log:
    r=subprocess.run([sys.executable,str(b/'pipeline/ingest_original_archives.py'),'--base',str(dest),'--source',sid],stdout=log,stderr=log)
   result={'source_id':sid,'returncode':r.returncode,'wall_seconds':time.perf_counter()-started}
   if r.returncode:raise RuntimeError((dest/(sid+'.log')).read_text()[-2000:])
   result['status']=json.loads((dest/'progress'/(sid+'.json')).read_text());return result
  start=time.perf_counter()
  with ThreadPoolExecutor(max_workers=workers) as pool:jobs=list(pool.map(run,['imdb','writingprompts']))
  elapsed=time.perf_counter()-start;db=connect(dest/'collection.sqlite3');n=counts(db)
  assert n=={'imdb':a.quota,'writingprompts':a.quota},n
  import gzip,hashlib
  for value in db.execute('SELECT row FROM passages'):
   row=json.loads(value[0]);raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]))
   assert raw['record']['text'][row['raw_start']:row['raw_end']]==row['text']
  digest=hashlib.sha256('\n'.join(x[0] for x in db.execute('SELECT id FROM passages ORDER BY id')).encode()).hexdigest();db.close()
  record={'trial':trial,'workers':workers,'wall_seconds':elapsed,'counts':n,'record_ids_sha256':digest,'jobs':jobs};records.append(record)
  atomic_json(root/'progress.json',{'trials':records});print(json.dumps(record),flush=True)
 assert len({r['record_ids_sha256'] for r in records})==1
 seq=statistics.mean(r['wall_seconds'] for r in records if r['workers']==1)
 par=statistics.mean(r['wall_seconds'] for r in records if r['workers']==2)
 result={'trials':records,'sequential_mean_seconds':seq,'parallel_mean_seconds':par,'speedup':seq/par,
 'recommended_concurrency':2 if par<seq*.95 else 1,'same_records_every_trial':True,'quota_per_source':a.quota,
 'scope':'End-to-end checksum, extraction/cache validation, parsing, SQLite dedup/checkpoint/original writes on cached source archives. Network download excluded; WritingPrompts cache is prepared and validated before all timed trials.',
 'warmup_seconds_excluded':warmup,'updated_at':now()}
 atomic_json(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
if __name__=='__main__':main()
