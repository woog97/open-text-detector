"""One quota-bounded ICLR chunk, with persistent timing and shared worker locks."""
import argparse,fcntl,json,os,shutil,sqlite3,time
from datetime import datetime
from pathlib import Path
from openreview_downloader import queue as q
PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'research/data/openreview_iclr2027_all'
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--chunk',required=True);a=parser.parse_args();assert a.chunk.isdigit()
 chunk=ROOT/'chunks'/a.chunk;chunk.mkdir(parents=True,exist_ok=True)
 locks=[]
 for p in [ROOT/'shared-account.lock',PROJECT/'research/data/openreview_legacy_download_queue/runner.lock',PROJECT/'research/data/reviewbench_download_queue/runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert (ROOT/'submissions-visible.json').exists() and (PROJECT/'research/data/openreview_legacy_download_queue/STOP_DIRECT').exists()
 queue=ROOT/'download_queue';db=q.connect(queue/'queue.sqlite3')
 quota=json.loads((ROOT/'quota-state.json').read_text());deadline=0
 headers=quota.get('last_response_headers',{})
 if headers.get('ratelimit-remaining')=='0' or headers.get('retry-after'):
  deadline=quota.get('server_reset_not_before',0)
 continuation=(chunk/'continue-current-window.json').exists()
 if continuation:
  headers=quota.get('last_response_headers',{})
  assert int(headers.get('ratelimit-remaining','0'))>0 and not headers.get('retry-after'), 'No confirmed unused allowance'
  deadline=0 # User-authorized continuation; reset is not a block while allowance remains.
 # Other known worker ledgers share quota; inspect all recent attachment attempts.
 recent=[]
 for p in [*(PROJECT/'research/data').glob('*/queue.sqlite3'),queue/'queue.sqlite3']:
  c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row
  try:
   for r in c.execute('SELECT * FROM attempts ORDER BY id DESC LIMIT 200'):
    started=datetime.fromisoformat(r['started_at']).timestamp()
    if started>time.time()-3600:recent.append(started)
   for r in c.execute("SELECT value FROM settings WHERE key='not_before'"):deadline=max(deadline,float(r['value']))
  except sqlite3.Error:pass
  finally:c.close()
 while time.time()<deadline:
  write(chunk/'waiting.json',{'not_before':deadline,'reason':'persisted actual start and shared/server limits'});time.sleep(min(30,deadline-time.time()))
 before={r[0] for r in db.execute("SELECT id FROM papers WHERE status IN ('downloaded','remote_verified')")};db.close()
 original=q.begin_attempt;attempts=0;first=None
 timing_path=chunk/'timing.json'
 if timing_path.exists():
  prior=json.loads(timing_path.read_text());first=prior['first_attachment_request_epoch']
  # Recover saved responses, then close the interrupted chunk without another request.
  db=q.connect(queue/'queue.sqlite3');q.recover(db,queue)
  saved=[dict(r) for r in db.execute('SELECT * FROM attempts') if datetime.fromisoformat(r['started_at']).timestamp()>=first]
  ids={fid for attempt in saved for fid in json.loads(attempt['ids'])}
  papers=[dict(r) for r in db.execute("SELECT * FROM papers WHERE status IN ('downloaded','remote_verified')") if r['id'] in ids]
  write(chunk/'manifest.json',{'chunk_id':a.chunk,'papers':papers,'downloads_finished_at':q.utc(),'download_stop_reason':'Recovered interrupted chunk; no additional attachment requests'})
  write(chunk/'request-receipt.json',{'attempts':saved,'settings':[dict(r) for r in db.execute('SELECT * FROM settings')]})
  return
 def begin(db,batch,papers):
  nonlocal first,attempts
  if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('4 GiB disk reserve reached; close partial chunk for processing')
  if attempts>=140:raise RuntimeError('Chunk request allowance reached')
  if continuation and attempts>=int(quota['last_response_headers']['ratelimit-remaining']):raise RuntimeError('Remaining server allowance reached')
  if first is None:
   first=time.time();write(timing_path,{'first_attachment_request_epoch':first,'first_attachment_request_at':q.utc(),'next_chunk_not_before':first+3600})
   if not continuation:
    quota['next_chunk_not_before']=first+3600;quota['first_attachment_request_at']=q.utc()
   write(ROOT/'quota-state.json',quota)
  attempts+=1
  aid=original(db,batch,papers)
  write(chunk/'request-count.json',{'reserved_attachment_requests':attempts,'other_recent_requests':len(recent),'last_attempt_id':aid})
  return aid
 q.begin_attempt=begin
 failure=None
 try:q.main(['--root',str(queue),'--max-batches','140','--stop-at-limit','--retry-failures'])
 except (Exception,SystemExit) as e:failure=str(e)
 finally:
  db=q.connect(queue/'queue.sqlite3');papers=[dict(r) for r in db.execute("SELECT * FROM papers WHERE status IN ('downloaded','remote_verified')") if r['id'] not in before]
  requests=[dict(r) for r in db.execute('SELECT * FROM attempts') if first is not None and datetime.fromisoformat(r['started_at']).timestamp()>=first]
  write(chunk/'manifest.json',{'chunk_id':a.chunk,'papers':papers,'downloads_finished_at':q.utc(),'download_stop_reason':failure})
  write(chunk/'request-receipt.json',{'attempts':requests,'settings':[dict(r) for r in db.execute('SELECT * FROM settings')]})
  for r in requests:
   h=json.loads(r['headers'] or '{}');finished=datetime.fromisoformat(r['finished_at'] or r['started_at']).timestamp()
   for k in ['retry-after','ratelimit-reset']:
    if str(h.get(k,'')).isdigit():quota['server_reset_not_before']=max(quota.get('server_reset_not_before',0),finished+int(h[k])+2)
  if requests:quota['last_response_headers']=json.loads(requests[-1]['headers'] or '{}')
  quota['requests_this_chunk']=len(requests);write(ROOT/'quota-state.json',quota)
  print(json.dumps({'chunk':a.chunk,'downloaded':len(papers),'requests':len(requests),'stop_reason':failure}),flush=True)
if __name__=='__main__':main()
