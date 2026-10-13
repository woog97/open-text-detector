"""Upload a validated chunk when extraction finishes; never delete local files."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'research/data/openreview_iclr2027_all'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--chunk',required=True);a=parser.parse_args();assert a.chunk.isdigit()
 os.chdir(ROOT);chunk=DATA/'chunks'/a.chunk
 guard=(chunk/'upload-supervisor.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(phase,**kw):
  p=chunk/'upload-state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'at':time.time(),**kw},indent=2));tmp.replace(p)
 status('waiting_for_validated_extraction')
 while True:
  ex=chunk/'extraction.json'
  if ex.exists() and json.loads(ex.read_text()).get('verified'):
   workflow=(chunk/'workflow.lock').open('a')
   try:fcntl.flock(workflow,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:workflow.close();time.sleep(15);continue
   if json.loads(ex.read_text()).get('verified'):break
   workflow.close()
  time.sleep(15)
 publication=(DATA/'pipeline.lock').open('a');fcntl.flock(publication,fcntl.LOCK_EX)
 status('uploading')
 result=subprocess.run([sys.executable,'scripts/publish_iclr2027_chunk.py','--chunk',a.chunk])
 status('uploaded' if result.returncode==0 else 'needs_repair',returncode=result.returncode,local_files_preserved=True)
if __name__=='__main__':main()
