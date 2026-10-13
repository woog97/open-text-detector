"""Consume validated immutable extraction microchunks; verify both remotes before cleanup."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'research/data/openreview_iclr2027_all'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 os.chdir(ROOT)
 lock=(DATA/'upload-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=DATA/'upload-monitor-state.json';failures={}
 while True:
  if (DATA/'STOP_UPLOADS').exists():
   write(state,{'pid':os.getpid(),'updated_at':time.time(),'phase':'stopped_by_user'});return
  ready=[]
  for c in sorted((DATA/'chunks').iterdir()):
   if not c.name.isdigit() or not (c/'extraction-ready.json').exists():continue
   if read(c/'cleanup-receipt.json').get('complete'):continue
   if not read(c/'extraction.json').get('verified'):continue
   if failures.get(c.name,0)>time.time():continue
   ready.append(c)
  write(state,{'pid':os.getpid(),'updated_at':time.time(),'phase':'waiting' if not ready else 'processing','ready_chunks':[c.name for c in ready]})
  for c in ready:
   if (DATA/'STOP_UPLOADS').exists():return
   workflow=(c/'workflow.lock').open('a');publication=(DATA/'pipeline.lock').open('a')
   try:
    fcntl.flock(workflow,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(publication,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:
    workflow.close();publication.close();continue
   try:
    write(state,{'pid':os.getpid(),'updated_at':time.time(),'phase':'publishing_and_verifying','chunk':c.name})
    with (c/'upload-cleanup.log').open('a') as log:
     cf=read(c/'cloudflare-receipt.json');hf=read(c/'hf-receipt.json')
     # Existing immutable readback receipts allow interrupted cleanup to resume without source PDFs.
     if not (cf.get('readback_verified') and cf.get('website_published') and hf.get('readback_verified')):
      subprocess.run([sys.executable,'scripts/publish_iclr2027_chunk.py','--chunk',c.name],stdout=log,stderr=subprocess.STDOUT,check=True)
     manifest=read(c/'manifest.json');parent=manifest.get('source_chunk') or manifest.get('parent_chunk')
     defer=bool(parent and not (DATA/'chunks'/str(parent)/'manifest.json').exists())
     cmd=[sys.executable,'scripts/verify_cleanup_iclr2027_chunk.py','--chunk',c.name]
     if defer:cmd.append('--verify-only')
     subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
     if defer:failures[c.name]=time.time()+60
    write(c/'upload-state.json',{'pid':os.getpid(),'at':time.time(),'phase':'verified_waiting_for_parent_close' if defer else 'verified_and_cleaned'})
   except Exception as e:
    failures[c.name]=time.time()+300
    write(c/'upload-state.json',{'pid':os.getpid(),'at':time.time(),'phase':'needs_repair','error':str(e),'retry_after':failures[c.name],'local_files_preserved_until_verified':True})
   finally:workflow.close();publication.close()
  time.sleep(15)
if __name__=='__main__':main()
