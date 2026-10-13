"""Extract/repair one immutable unit, sharing two global two-worker slots."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from process_new_positioned_papers import write
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'research/data/openreview_iclr2027_all'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 unit=DATA/'chunks'/sys.argv[1];os.chdir(ROOT)
 lock=(unit/'extraction-stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (unit/'extraction-ready.json').exists():return
 # Let pre-transition pipeline/repair finish without interruption.
 workflow=(unit/'workflow.lock').open('a');fcntl.flock(workflow,fcntl.LOCK_EX);fcntl.flock(workflow,fcntl.LOCK_UN)
 slot=None
 while slot is None:
  for i in range(2):
   f=(DATA/f'extraction-slot-{i}.lock').open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);slot=f;break
   except BlockingIOError:f.close()
  if slot is None:time.sleep(5)
 env=dict(os.environ,ICLR_EXTRACTION_WORKERS='2')
 with (unit/'pipeline.log').open('a') as log:
  ex=json.loads((unit/'extraction.json').read_text()) if (unit/'extraction.json').exists() else {}
  if not ex.get('verified'):
   subprocess.run([sys.executable,'scripts/run_iclr2027_pipeline.py','--chunk',unit.name,'--download-extract-only'],stdout=log,stderr=subprocess.STDOUT,env=env)
   ex=json.loads((unit/'extraction.json').read_text())
   if not ex.get('verified'):subprocess.run([sys.executable,'scripts/repair_iclr2027_chunk.py','--chunk',unit.name,'--extract-only'],stdout=log,stderr=subprocess.STDOUT,env=env)
 ex=json.loads((unit/'extraction.json').read_text());m=json.loads((unit/'manifest.json').read_text());ids={r['id'] for r in m['papers']}
 if not ex.get('verified') or ids!={p['forum_id'] for p in ex['papers']}:raise RuntimeError('Extraction needs repair; preserve PDFs')
 for p in ex['papers']:
  assert not p['verification']['issues'];assert digest(ROOT/p['text_file'])==p['blob_sha256']
 write(unit/'extraction-ready.json',{'verified':True,'source_chunk':m['source_chunk'],'papers':len(ids),'manifest_sha256':digest(unit/'manifest.json'),'extraction_sha256':digest(unit/'extraction.json'),'verified_at':time.time()})
if __name__=='__main__':main()
