"""Remote-only bounded collection after the user-approved quota reallocation."""
from pathlib import Path
import json,sys,subprocess,os,time,sqlite3,hashlib,fcntl
from huggingface_hub import HfApi
from collect_pool import atomic_json,now
B=Path('/tmp/pangram-human-active-20261002');ROOT=B/'finish100k';P=B/'pipeline'
token=sys.stdin.readline().strip();assert token
api=HfApi(token=token);bucket='open-text-detector/training-storage';prefix='workspace/human-source-mix-v2-recovered-20261002/'

def checkpoint():
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());path=B/'checkpoints'/('finish100k-active-'+stamp+'.sqlite3')
 s=sqlite3.connect(B/'collection.sqlite3');s.execute('BEGIN');s.execute('SELECT count(*) FROM passages').fetchone();d=sqlite3.connect(path);s.backup(d,pages=4096);s.close();d.execute('PRAGMA journal_mode=DELETE');assert d.execute('PRAGMA quick_check').fetchall()==[('ok',)];counts=dict(d.execute('SELECT source,count(*) FROM passages GROUP BY source'));d.close()
 key=prefix+'checkpoints/'+path.name;api.batch_bucket_files(bucket,add=[(path,key)]);items=list(api.get_bucket_paths_info(bucket,[key]));assert len(items)==1 and items[0].size==path.stat().st_size
 r={'database':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bucket_key':key,'candidate_total':sum(counts.values()),'counts':counts,'verification':'committed size metadata; full GET pending final release','at':now()};atomic_json(ROOT/'checkpoint.json',r);api.batch_bucket_files(bucket,add=[(ROOT/'checkpoint.json',prefix+'finish100k/checkpoint.json')]);print('CHECKPOINT',json.dumps(r),flush=True)

with (ROOT/'supervisor.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs={'gutenberg':[sys.executable,str(P/'collect_stream_source.py'),'--base',str(B),'--source','gutenberg'], 'foodista':[sys.executable,str(P/'collect_stream_source.py'),'--base',str(B),'--source','foodista'], 'asap2':[sys.executable,str(P/'ingest_asap2.py'),'--base',str(B)], 'voa':[sys.executable,str(P/'finish_voa.py')]}
 children={};logs=[]
 try:
  for sid,cmd in jobs.items():
   with (B/(sid+'.lock')).open('a') as source_lock:fcntl.flock(source_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   log=(ROOT/(sid+'.log')).open('a');logs.append(log);child=subprocess.Popen(cmd,cwd=P,stdout=log,stderr=log,start_new_session=True,env=dict(os.environ));children[sid]=child
   atomic_json(ROOT/(sid+'-process.json'),{'pid':child.pid,'start_ticks':Path('/proc',str(child.pid),'stat').read_text().split()[21],'source':sid,'started_at':now()})
  last=time.monotonic()
  while any(c.poll() is None for c in children.values()):
   time.sleep(5);atomic_json(ROOT/'status.json',{'state':'collecting','workers':{s:c.poll() for s,c in children.items()},'at':now()})
   if time.monotonic()-last>=120:checkpoint();last=time.monotonic()
  checkpoint();atomic_json(ROOT/'status.json',{'state':'workers_finished','workers':{s:c.returncode for s,c in children.items()},'at':now()})
 except BaseException:
  for c in children.values():
   if c.poll() is None:c.terminate()
  for c in children.values():
   if c.poll() is None:c.wait(timeout=30)
  atomic_json(ROOT/'status.json',{'state':'failed_children_stopped','at':now()});raise
 finally:
  for l in logs:l.close()
