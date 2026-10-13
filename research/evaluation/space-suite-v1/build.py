"""Package existing evaluations and the reserved manuscripts; no model calls."""
from pathlib import Path
import json,gzip,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent/'package'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 OUT.mkdir(exist_ok=True);src=ROOT/'benchmarks/otd/eval_suite/bundle';manifest={'version':1,'profiles':{},'notes':'Profiles overlap; never sum them. Assistance is diagnostic only. Full manuscripts have document labels, not invented token gold.'}
 for name in ['workflow','comparison','assistance','full','context']:
  rows=[json.loads(l) for l in gzip.open(src/(name+'.jsonl.gz'),'rt')]
  # The assistance source also contains calibration examples. Evaluate test only.
  if name=='assistance':rows=[r for r in rows if r.get('split')=='test']
  for r in rows:r.setdefault('granularity','character_provenance' if 'regions' in r else 'native_document_label')
  write(name,rows,manifest)
 reserved=ROOT/'research/evaluation/full-paper-holdout-20261003'
 ids={r['source_paper_id']:r for r in map(json.loads,(reserved/'eval.jsonl').read_text().splitlines())}
 rows=[]
 for r in map(json.loads,(ROOT/'research/data/synthetic-papers-600-hf-20261003/papers.jsonl').read_text().splitlines()):
  if r['source_paper_id'] not in ids:continue
  assert r['paper_sha256']==ids[r['source_paper_id']]['paper_sha256']
  text=r['paper_markdown'];rows.append({'id':'full-paper/'+str(r['paper_id']),'text':text,'text_sha256':sha(text.encode()),'dataset':'full_manuscripts','group_id':r['source_paper_id'],'generator':r['final_revision_model'],'label':'ai','granularity':'document_native_label','split':'test','domain':'research_paper','human_supplied_sections':r['human_supplied_sections']})
 assert len(rows)==120
 from collections import Counter
 assert set(Counter(r['generator'] for r in rows).values())=={40}
 write('manuscripts',rows,manifest)
 for name in ['common.py','score_wide_eval.py']:shutil.copy2(src/name,OUT/name)
 for name in ['run_eval.py']:shutil.copy2(Path(__file__).parent/name,OUT/name)
 shutil.copy2(reserved/'reservation.json',OUT/'training-exclusions.json')
 manifest['files']={p.name:sha(p.read_bytes()) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['profiles'],indent=2))
def write(name,rows,m):
 blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(OUT/(name+'.jsonl.gz')).write_bytes(gzip.compress(blob,mtime=0));m['profiles'][name]={'rows':len(rows),'sha256':sha(blob)}
if __name__=='__main__':main()
