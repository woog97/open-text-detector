"""Remove only unchanged PDF/PGF sources after verified R2 and Space transfers."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/iclr-round2-upload-20261003'
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());done=json.loads((OUT/'completion.json').read_text());assert done['complete'] and done['papers']==6949
 receipts={r['path']:r for r in done['files']};selected={p[k] for p in manifest['papers'] for k in ['pdf','text_file']};assert len(selected)==13898
 removed=[];missing=[];changed=[]
 log=OUT/'local-deletion-receipts.jsonl'
 with log.open('a') as out:
  for rel in sorted(selected):
   p=ROOT/rel;r=receipts[rel]
   assert p.suffix in ['.pdf','.pgf'] and p.resolve().is_relative_to((ROOT/'research').resolve()) and not p.is_symlink()
   assert all(part['verified'] for part in r['parts'])
   if not p.exists():missing.append(rel);continue
   stat=p.stat();sha=hashlib.sha256(p.read_bytes()).hexdigest()
   if sha!=r['sha256'] or stat.st_size!=r['bytes'] or p.stat().st_mtime_ns!=stat.st_mtime_ns:changed.append(rel);continue
   p.unlink();row={'path':rel,'sha256':sha,'bytes':stat.st_size,'allocated_bytes':stat.st_blocks*512,'deleted_at':time.time()};removed.append(row);out.write(json.dumps(row)+'\n');out.flush()
   if len(removed)%1000==0:print(f'Deleted {len(removed)} verified files',flush=True)
 summary={'deleted_files':len(removed),'freed_bytes':sum(r['bytes'] for r in removed),'freed_allocated_bytes':sum(r['allocated_bytes'] for r in removed),'already_absent':len(missing),'changed_preserved':changed,'completed_at':time.time()}
 (OUT/'local-deletion-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
