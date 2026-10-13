"""Space-local approved Wikipedia continuation with versioned bucket checkpoints."""
from pathlib import Path
import fcntl,hashlib,json,os,sqlite3,subprocess,sys,time,requests
from urllib.parse import quote
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
from collect_pool import atomic_json,now

disable_progress_bars()
b=Path('/tmp/pangram-human-active-20261002');bucket='open-text-detector/training-storage';prefix='workspace/human-source-mix-v2-recovered-20261002/'
token=sys.stdin.readline().strip();assert token;api=HfApi(token=token)
progress=b/'progress';progress.mkdir(exist_ok=True)
root=b/'healthy-wikipedia';root.mkdir(exist_ok=True)
with (root/'supervisor.lock').open('a') as supervisor_lock:
 fcntl.flock(supervisor_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Ensure no existing source collector before spawning. The source child holds this lock itself.
 with (b/'wikipedia.lock').open('a') as source_lock:fcntl.flock(source_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 processfile=root/'worker-process.json'
 if processfile.exists():
  previous=json.loads(processfile.read_text());stat=Path('/proc',str(previous['pid']),'stat')
  if stat.exists() and stat.read_text().split()[21]==previous['start_ticks'] and stat.read_text().split()[2]!='Z':raise RuntimeError('Worker still alive')
 def checkpoint(verify_bytes=False):
  name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());target=b/'checkpoints'/('wikipedia-'+name+'.sqlite3')
  assert not target.exists()
  live=sqlite3.connect(b/'collection.sqlite3',timeout=60)
  # Pin a WAL read snapshot so concurrent commits cannot repeatedly restart backup.
  live.execute('BEGIN');live.execute('SELECT count(*) FROM passages').fetchone()
  out=sqlite3.connect(target);live.backup(out,pages=4096);live.close();out.execute('PRAGMA journal_mode=DELETE')
  assert out.execute('PRAGMA quick_check').fetchall()==[('ok',)]
  counts=dict(out.execute('SELECT source,count(*) FROM passages GROUP BY source'));out.close()
  quota={x['source_id']:x['planned_passages'] for x in json.loads((b/'pipeline/sampling-plan.json').read_text())['source_quotas']};assert all(n<=quota[s] for s,n in counts.items())
  h=hashlib.sha256(target.read_bytes()).hexdigest();key=prefix+'checkpoints/'+target.name
  api.batch_bucket_files(bucket,add=[(target,key)]);objects=list(api.get_bucket_paths_info(bucket,[key]));assert len(objects)==1 and objects[0].size==target.stat().st_size
  if verify_bytes:
   response=requests.get('https://huggingface.co/buckets/'+bucket+'/resolve/'+quote(key,safe=''),headers={'Authorization':'Bearer '+token,'Accept-Encoding':'identity'},stream=True,timeout=(10,90));response.raise_for_status();remote_hash=hashlib.sha256()
   for chunk in response.iter_content(4*1024*1024):remote_hash.update(chunk)
   assert remote_hash.hexdigest()==h,'Full checkpoint readback mismatch'
  result={'remote_full_checksum_readback':verify_bytes,'database':str(target),'bucket_key':key,'sha256':h,'xet_hash':objects[0].xet_hash,'bytes':objects[0].size,'source_counts':counts,'candidate_total':sum(counts.values()),'admitted_total':0,'updated_at':now()}
  body=(json.dumps(result,indent=2)+'\n').encode();api.batch_bucket_files(bucket,add=[(body,prefix+'checkpoints/'+name+'-receipt.json'),(body,prefix+'latest-checkpoint.json')]);atomic_json(root/'checkpoint.json',result)
  print(json.dumps({'checkpoint':key,'candidate_total':sum(counts.values()),'wikipedia':counts.get('wikipedia',0)}),flush=True)
 with (root/'worker.log').open('a') as log:
  child=subprocess.Popen([sys.executable,str(b/'pipeline/collect_stream_source.py'),'--base',str(b),'--source','wikipedia'],cwd=b/'pipeline',stdout=log,stderr=log,env={**os.environ,'HF_HOME':'/tmp/pangram-human-hf-cache'},start_new_session=True)
 identity={'pid':child.pid,'start_ticks':Path('/proc',str(child.pid),'stat').read_text().split()[21],'source_id':'wikipedia','state':'running','started_at':now()};atomic_json(processfile,identity)
 last=time.monotonic()-120;first_checkpoint=True
 try:
  while child.poll() is None:
   time.sleep(5)
   if time.monotonic()-last>=120:checkpoint(verify_bytes=first_checkpoint);first_checkpoint=False;last=time.monotonic()
  checkpoint(verify_bytes=True);atomic_json(root/'status.json',{'state':'finished' if child.returncode==0 else 'failed','returncode':child.returncode,'updated_at':now()})
 except BaseException:
  # Only our verified child is interrupted if durable saving fails; never replay paid calls (none here).
  if child.poll() is None:child.terminate();child.wait(timeout=30)
  atomic_json(root/'status.json',{'state':'checkpoint_failed_worker_stopped','updated_at':now()});raise
