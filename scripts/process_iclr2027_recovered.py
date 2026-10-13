"""Extract recovered batch immediately; publish after the preceding chunk clears."""
import concurrent.futures,json,fcntl,subprocess,sys,time,os
from pathlib import Path
from run_iclr2027_pipeline import extract_verify
from process_new_positioned_papers import write
root=Path('research/data/openreview_iclr2027_all');chunk=root/'chunks/000003'
def main():
 lock=(chunk/'processing.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=json.loads((chunk/'manifest.json').read_text());results=[];errors=[]
 papers=[{'pdf':str(Path(p['file']).relative_to(Path.cwd())),'pdf_sha256':p['sha256'],'forum_id':p['id']} for p in manifest['papers']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  tasks={pool.submit(extract_verify,p):p for p in papers}
  for future in concurrent.futures.as_completed(tasks):
   try:r=future.result();results.append(r)
   except Exception as e:errors.append({'id':tasks[future]['forum_id'],'error':str(e)})
   write(chunk/'extraction.json',{'papers':results,'errors':errors,'verified':False})
 errors += [{'id':p['forum_id'],'issues':p['verification']['issues']} for p in results if p['verification']['issues']]
 write(chunk/'extraction.json',{'papers':results,'errors':errors,'verified':not errors and len(results)==len(papers)})
 print(json.dumps({'extracted':len(results),'errors':errors}),flush=True)
 fcntl.flock(lock,fcntl.LOCK_UN)
 if errors:return
 while True:
  if (root/'chunks/000002/cleanup-receipt.json').exists() and json.loads((root/'chunks/000002/cleanup-receipt.json').read_text()).get('complete'):break
  state=json.loads((root/'runner-state.json').read_text())
  if state['phase']=='needs_repair':return
  time.sleep(30)
 while True:
  f=(root/'pipeline.lock').open('a')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);f.close();break
  except BlockingIOError:f.close();time.sleep(10)
 subprocess.run([sys.executable,'scripts/run_iclr2027_pipeline.py','--chunk','000003'],check=True)
if __name__=='__main__':main()
