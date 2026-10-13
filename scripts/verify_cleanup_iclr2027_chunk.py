import json,hashlib,gzip,subprocess,time,sqlite3,fcntl,sys,concurrent.futures,os
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--chunk',required=True);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args();assert args.chunk.isdigit()
root=Path('research/data/openreview_iclr2027_all').resolve();chunk=root/'chunks'/args.chunk
lock=(chunk/'processing.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,'scripts');from process_new_positioned_papers import write
sys.path.insert(0,'backend');from open_text_detector_backend.result_codec import decode
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url):return subprocess.run(['curl','--fail','--silent','--show-error','--max-time','90','--retry','3','--retry-delay','2','--retry-all-errors',url],capture_output=True,check=True).stdout
hf=json.loads((chunk/'hf-receipt.json').read_text());cf=json.loads((chunk/'cloudflare-receipt.json').read_text());manifest=json.loads((chunk/'manifest.json').read_text());ex=json.loads((chunk/'extraction.json').read_text());assert hf['readback_verified'] and hf['private'] and cf['readback_verified'] and ex['verified']
# Exact coverage is required; legacy remote_verified status alone is never a cleanup gate.
from iclr2027_cleanup_gates import validate_coverage,authorize_local_path
ids=validate_coverage(manifest,ex,hf,cf)
base='https://open-text-detector.woog09.workers.dev/backend/v1/pdf-reader'
listed=json.loads(fetch(base+'?collection=iclr%2F2027'));assert all(r['site_id'] in {x['id'] for x in listed['items']} for r in cf['objects'])
worker_receipts=chunk/'worker-objects';worker_receipts.mkdir(exist_ok=True)
def verify_worker_object(ref):
 proof=worker_receipts/(ref['id']+'.json')
 if proof.exists() and json.loads(proof.read_text()).get('ref')==ref:return
 raw=fetch(base+'/'+ref['site_id'])
 if raw.startswith(b'\x1f\x8b'):assert sha(raw)==ref['detail_sha256'];raw=gzip.decompress(raw)
 d=json.loads(raw);assert d['forum_id']==ref['id'] and sha(d['text'].encode())==ref['text_sha256']
 p=next(p for p in ex['papers'] if p['forum_id']==ref['id']);source=decode(Path(p['text_file']).read_bytes())
 assert d['text']==source['text'] and d['position_maps'][ref['text_sha256']]=={'pages':source['pages'],'rectangles':source['rectangles']}
 pdf=fetch(base+'/'+ref['site_id']+'/file');assert len(pdf)==ref['pdf_bytes'] and sha(pdf)==ref['pdf_sha256']
 write(proof,{'ref':ref,'worker_verified_at':time.time(),'text_positions_and_pdf_verified':True})
with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(12,int(os.environ.get('ICLR_VERIFY_WORKERS','8'))))) as pool:
 list(pool.map(verify_worker_object,cf['objects']))
cf.update(worker_verified=True,worker_url=base,worker_verified_at=time.time());write(chunk/'cloudflare-receipt.json',cf)
receipt={'chunk_manifest_sha256':sha((chunk/'manifest.json').read_bytes()),'hf':hf,'cloudflare':cf,'extraction_verified':True,'verified_at':time.time()};write(chunk/'verification-receipt.json',receipt)
if args.verify_only:
 print(json.dumps({'verified':len(ids),'cleanup_deferred':True}));sys.exit(0)
# Persist remote completion BEFORE authorized deletion; reconciliation must preserve this status.
db=sqlite3.connect(root/'download_queue/queue.sqlite3')
with db:
 db.execute('CREATE TABLE IF NOT EXISTS remote_completion(id TEXT PRIMARY KEY,receipt TEXT,verified_at REAL)')
 for p in manifest['papers']:
  db.execute('INSERT OR REPLACE INTO remote_completion VALUES(?,?,?)',(p['id'],str(chunk/'verification-receipt.json'),time.time()))
  db.execute("UPDATE papers SET status='remote_verified' WHERE id=?",(p['id'],))
cleanup=[]
for p in manifest['papers']:
 path=authorize_local_path(p,root/'download_queue/pdfs')
 if path.exists():
  raw=path.read_bytes();assert len(raw)==p['bytes'] and sha(raw)==p['sha256'];path.unlink()
 cleanup.append({'id':p['id'],'path':str(path),'sha256':p['sha256'],'bytes':p['bytes'],'deleted':True})
 write(chunk/'cleanup-receipt.json',{'verification_receipt_sha256':sha((chunk/'verification-receipt.json').read_bytes()),'papers':cleanup,'complete':len(cleanup)==len(manifest['papers'])})
s=json.loads((root/'workflow-state.json').read_text());s.update(verified_remote_papers=db.execute('SELECT count(*) FROM remote_completion').fetchone()[0],local_pdf_cleanup_complete_for_chunk=args.chunk,publication_complete=False);write(root/'workflow-state.json',s)
print(json.dumps({'verified_and_cleaned':len(cleanup),'hf_commit':hf['commit']}))
