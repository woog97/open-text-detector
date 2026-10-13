"""Durable one-chunk pipeline: download, extract, publish, verify, gated cleanup."""
import argparse,concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from process_new_positioned_papers import extract_one,verify_one,write
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'research/data/openreview_iclr2027_all'
def extract_verify(paper):
 paper.update(extract_one(paper));paper['verification']=verify_one(paper);return paper

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--chunk',required=True);parser.add_argument('--overlap-download',action='store_true');parser.add_argument('--download-extract-only',action='store_true');a=parser.parse_args()
 if a.download_extract_only:a.overlap_download=True
 assert a.chunk.isdigit()
 os.chdir(ROOT)
 chunk=DATA/'chunks'/a.chunk;chunk.mkdir(parents=True,exist_ok=True)
 lock=(chunk/'workflow.lock' if a.overlap_download else DATA/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(phase,**extra):
  write(chunk/'runner-state.json' if a.overlap_download else DATA/'runner-state.json',{'pid':os.getpid(),'chunk':a.chunk,'phase':phase,'updated_at':time.time(),**extra});print(json.dumps({'phase':phase,**extra}),flush=True)
 try:
  if (chunk/'cleanup-receipt.json').exists() and json.loads((chunk/'cleanup-receipt.json').read_text()).get('complete'):
   status('chunk_complete',reason='Previously verified and cleaned; no work repeated');return
  if not (chunk/'manifest.json').exists():
   status('downloading');subprocess.run(['/tmp/openreview-benchmark-venv/bin/python','research/tools/openreview_downloader/run_iclr2027_chunk.py','--chunk',a.chunk],check=True)
  manifest=json.loads((chunk/'manifest.json').read_text())
  if not manifest['papers']:
   status('waiting',reason=manifest.get('download_stop_reason') or 'No new PDFs downloaded');return
  status('extracting',papers=len(manifest['papers']))
  previous=json.loads((chunk/'extraction.json').read_text()) if (chunk/'extraction.json').exists() else {'papers':[]}
  results={p['forum_id']:p for p in previous['papers'] if not p.get('verification',{}).get('issues') and p.get('verification')}
  pending=[{'pdf':str(Path(r['file']).relative_to(ROOT)),'pdf_sha256':r['sha256'],'forum_id':r['id']} for r in manifest['papers'] if r['id'] not in results]
  errors=[]
  with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get("ICLR_EXTRACTION_WORKERS", "4"))) as pool:
   tasks={pool.submit(extract_verify,p):p for p in pending}
   for future in concurrent.futures.as_completed(tasks):
    paper=tasks[future]
    try:
     result=future.result();results[paper['forum_id']]=result
     if result['verification']['issues']:errors.append({'id':paper['forum_id'],'issues':result['verification']['issues']})
    except Exception as e:errors.append({'id':paper['forum_id'],'error':str(e)})
    write(chunk/'extraction.json',{'papers':list(results.values()),'errors':errors,'verified':False})
    if len(results)%50==0:status('extracting',complete=len(results),total=len(manifest['papers']),errors=len(errors))
  good=len(results)==len(manifest['papers']) and not errors
  write(chunk/'extraction.json',{'papers':list(results.values()),'errors':errors,'verified':good})
  if not good:raise RuntimeError('Extraction validation needs repair; all PDFs retained')
  if a.download_extract_only:
   status('extraction_complete',papers=len(results));return
  if a.overlap_download:
   status('waiting_for_publication',papers=len(results))
   publication_lock=(DATA/'pipeline.lock').open('a')
   fcntl.flock(publication_lock,fcntl.LOCK_EX)
  status('publishing',papers=len(results));subprocess.run([sys.executable,'scripts/publish_iclr2027_chunk.py','--chunk',a.chunk],check=True)
  status('verifying_and_cleaning',papers=len(results));subprocess.run([sys.executable,'scripts/verify_cleanup_iclr2027_chunk.py','--chunk',a.chunk],check=True)
  status('chunk_complete',papers=len(results))
 except Exception as e:
  status('needs_repair',error=str(e));raise
if __name__=='__main__':main()
