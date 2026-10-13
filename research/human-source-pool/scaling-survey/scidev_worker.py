"""Isolated, bounded capacity audit; retain gates and source originals."""
import sys,os,time,json,gzip,sqlite3,hashlib,fcntl
from pathlib import Path
from collections import defaultdict,Counter
from urllib.parse import quote
import requests
from huggingface_hub import HfApi
from ingest_scidev import source_url,slug,retrieve,PublisherPacer,matching_regions,pairs
from expand_pool import connect,add
from collect_pool import atomic_json,now
b=Path(sys.argv[1]);token=sys.stdin.readline().strip();assert token
api=HfApi(token=token);bucket='open-text-detector/training-storage';prefix='workspace/human-source-mix-v2-recovered-20261002/scaling/scidev-v1/'
start=time.monotonic();deadline=start+2700;counts=Counter();errors=[];uploaded=set();position=0
lock=(b/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
db=connect(b/'stage.sqlite3');groups=defaultdict(list)
for line in gzip.open(b/'source-downloads/historical-wrappers.jsonl.gz','rt'):
 w=json.loads(line);url=source_url(w)
 if url:groups[slug(url)].append(w)
pacer=PublisherPacer(b)
def status(state):
 n=db.execute('select count(*) from passages').fetchone()[0]
 result={'source_id':'scidev','state':state,'count':n,'baseline':926,'additional_candidates':n-926,'target_bound':9910,'position':position,'article_groups':len(groups),'length_counts':dict(db.execute('select bin,count(*) from passages group by bin')),'reasons':dict(counts),'errors':errors[-10:],'all_quarantined':True,'production_unchanged':True,'global_production_dedup_pending':True,'started_at':started,'updated_at':now(),'max_seconds':2700}
 atomic_json(b/'status.json',result);return result
started=now()
def checkpoint(final=False):
 status('checkpointing');snapshot=b/'checkpoints'/('stage-'+str(time.time_ns())+'.sqlite3');snapshot.parent.mkdir(exist_ok=True)
 db.commit();out=sqlite3.connect(snapshot);db.backup(out);out.execute('PRAGMA journal_mode=DELETE');assert out.execute('pragma integrity_check').fetchone()==('ok',);out.close()
 files=[p for p in (b/'source-downloads').rglob('*') if p.is_file() and str(p) not in uploaded]+[p for p in (b/'pipeline').glob('*.py') if str(p) not in uploaded]+[snapshot,b/'status.json']
 api.batch_bucket_files(bucket,add=[(p,prefix+str(p.relative_to(b))) for p in files]);uploaded.update(map(str,files))
 key=prefix+str(snapshot.relative_to(b));info=list(api.get_bucket_paths_info(bucket,[key]));assert info[0].size==snapshot.stat().st_size
 sha=hashlib.sha256(snapshot.read_bytes()).hexdigest();verified=False
 if final or not (b/'checkpoint.json').exists():
  r=requests.get('https://huggingface.co/buckets/'+bucket+'/resolve/'+quote(key,safe=''),headers={'Authorization':'Bearer '+token,'Accept-Encoding':'identity'},stream=True,timeout=(15,90));r.raise_for_status();h=hashlib.sha256()
  for chunk in r.iter_content(4*1024*1024):h.update(chunk)
  assert h.hexdigest()==sha;verified=True
 receipt={'bucket':bucket,'key':key,'sha256':sha,'bytes':snapshot.stat().st_size,'full_get_verified':verified,'candidate_count':db.execute('select count(*) from passages').fetchone()[0],'originals_uploaded':len(uploaded),'updated_at':now()};atomic_json(b/'checkpoint.json',receipt);api.batch_bucket_files(bucket,add=[(b/'checkpoint.json',prefix+'checkpoint.json')]);return receipt
try:
 checkpoint();last=time.monotonic()
 # Longest captures first; validate ALL alternative captures, not arbitrary first hit.
 ordered=sorted(groups.items(),key=lambda kv:-max(len(w['record']['text'].split()) for w in kv[1]))
 for position,(article,wrappers) in enumerate(ordered,1):
  if time.monotonic()>deadline:break
  if db.execute('select count(*) from passages').fetchone()[0]>=9910:break
  if db.execute("select 1 from cursors where source='scidev-capacity-v1' and file=?",(article,)).fetchone():continue
  # Existing document families are never sampled twice across differing capture texts.
  if db.execute('select 1 from passages where doc=?',('scidev:'+article,)).fetchone():counts['existing_family_preserved']+=1;continue
  try:
   ev,manifest=retrieve(b,wrappers[0],pacer)
   options=[]
   for w in wrappers:
    regions,overlap=matching_regions(w['record']['text'],ev['current_article_body'])
    score=max((len(w['record']['text'][a:z].split()) for a,z in regions),default=0)
    options.append((score,overlap.get('matched_tokens',0),w))
   _,_,best=max(options,key=lambda x:(x[0],x[1]));rows=list(pairs(best,ev,manifest,[9910]*4))
   counts['matched_article' if rows else 'no_eligible_spans']+=1
   with db:
    for row,raw in rows:
     assert row['text']==raw['record']['text'][row['raw_start']:row['raw_end']]
     counts['new_passages']+=int(add(db,row,raw,9910))
    db.execute("insert or replace into cursors values('scidev-capacity-v1',?,1,1)",(article,))
  except Exception as exc:
   counts['rejected_or_network_error']+=1;errors.append({'article':article,'error':str(exc)[:180]})
   if pacer.consecutive_limits>=3:break
  status('running')
  if time.monotonic()-last>=180:checkpoint();last=time.monotonic()
 state='completed_cached_capacity_scan' if position==len(ordered) else ('publisher_rate_limited' if pacer.consecutive_limits>=3 else 'bounded_run_checkpointed')
 checkpoint(final=True);result=status(state);api.batch_bucket_files(bucket,add=[(b/'status.json',prefix+'status.json')]);print(json.dumps(result),flush=True)
except Exception as exc:
 errors.append({'error':str(exc)[:300]});status('failed_preserved');raise
