"""Extract a frozen snapshot while downloads continue; serialize parent handoff."""
import concurrent.futures,fcntl,json,os,signal,sqlite3,sys,time
from pathlib import Path
from process_new_positioned_papers import extract_one,verify_one,write
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'research/data/openreview_iclr2027_all'
def extract(p):
 p.update(extract_one(p));p['verification']=verify_one(p);return p

def main():
 import argparse,subprocess
 parser=argparse.ArgumentParser();parser.add_argument('--chunk',required=True);parser.add_argument('--parent-pid',type=int,required=True);a=parser.parse_args()
 os.chdir(ROOT);chunk=DATA/'chunks'/a.chunk
 lock=(chunk/'live-extraction.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 command=subprocess.check_output(['ps','-p',str(a.parent_pid),'-o','command='],text=True)
 assert 'run_iclr2027_pipeline.py' in command and '--chunk '+a.chunk in command
 os.kill(a.parent_pid,signal.SIGSTOP) # Downloader child continues; parent cannot race extraction.
 try:
  state=json.loads((chunk/'runner-state.json').read_text());assert state['phase']=='downloading'
  first=json.loads((chunk/'timing.json').read_text())['first_attachment_request_at']
  db=sqlite3.connect(DATA/'download_queue/queue.sqlite3');db.row_factory=sqlite3.Row
  ids={fid for row in db.execute('SELECT ids FROM attempts WHERE started_at>=?',(first,)) for fid in json.loads(row[0])}
  rows=[dict(r) for r in db.execute("SELECT * FROM papers WHERE status='downloaded'") if r['id'] in ids];db.close()
  write(chunk/'live-extraction-snapshot.json',{'papers':rows,'at':time.time()})
  previous=json.loads((chunk/'extraction.json').read_text()) if (chunk/'extraction.json').exists() else {'papers':[]}
  results={p['forum_id']:p for p in previous['papers']};errors=[]
  pending=[{'forum_id':r['id'],'pdf':str(Path(r['file']).relative_to(ROOT)),'pdf_sha256':r['sha256']} for r in rows if r['id'] not in results]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(extract,p):p['forum_id'] for p in pending}
   for f in concurrent.futures.as_completed(futures):
    fid=futures[f]
    try:
     p=f.result();results[fid]=p
     if p['verification']['issues']:errors.append({'id':fid,'issues':p['verification']['issues']})
    except Exception as e:errors.append({'id':fid,'error':str(e)})
    write(chunk/'extraction.json',{'papers':list(results.values()),'errors':errors,'verified':False})
    if len(results)%25==0:
     s={'pid':os.getpid(),'complete':len(results),'snapshot_total':len(rows),'errors':len(errors),'at':time.time()};write(chunk/'live-extraction-state.json',s);print(json.dumps(s),flush=True)
 finally:
  os.kill(a.parent_pid,signal.SIGCONT)
if __name__=='__main__':main()
