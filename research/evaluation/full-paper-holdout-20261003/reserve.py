"""Create an ID-only, reproducible evaluation reservation; preserve source files."""
from pathlib import Path
import json,hashlib,collections
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'research/data/synthetic-papers-600-hf-20261003/papers.jsonl'
blob=SRC.read_bytes();rows=[json.loads(l) for l in blob.splitlines()]
assert len(rows)==600
assert len({r['source_paper_id'] for r in rows})==600
assert len({r['paper_sha256'] for r in rows})==600
by=collections.defaultdict(list)
for r in rows:by[r['final_revision_model']].append(r)
assert set(by)=={'gpt-6.1-sol','gpt-6-sol','gpt-6-luna'}
salt='full-paper-evaluation-reservation-20261003-v1'
held=set()
for model,group in by.items():
 ordered=sorted(group,key=lambda r:hashlib.sha256((salt+'/'+r['source_paper_id']).encode()).hexdigest())
 assert len(ordered)>=40
 held.update(r['source_paper_id'] for r in ordered[:40])
assignments=[]
for r in rows:
 assignments.append({**{k:r[k] for k in ['paper_id','seed_id','source_paper_id','paper_sha256','final_revision_model','writer_model','models_used','source_batch','markdown_path']},'split':'evaluation' if r['source_paper_id'] in held else 'train','training_allowed':r['source_paper_id'] not in held})
for name,rs in [('assignments',assignments),('eval',[r for r in assignments if r['split']=='evaluation']),('train',[r for r in assignments if r['split']=='train'])]:
 p=OUT/(name+'.jsonl');b=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs).encode()
 if p.exists():assert p.read_bytes()==b,'Refusing to alter an existing reservation'
 else:p.write_bytes(b)
report={'source':str(SRC.relative_to(ROOT)),'source_sha256':hashlib.sha256(blob).hexdigest(),'selection':'40 per final_revision_model, stable source-ID hash order','salt':salt,'counts':{m:{'total':len(g),'evaluation':40,'train':len(g)-40} for m,g in sorted(by.items())},'evaluation_rows':120,'train_rows':480,'exclude_from_training':{'source_paper_ids':sorted(held),'paper_sha256':sorted(r['paper_sha256'] for r in rows if r['source_paper_id'] in held)},'policy':'All excerpts, revisions and alternate views of reserved papers stay out of training. Downstream preparation must consume these exclusions. Existing source dataset storage split named train is superseded by this sidecar for this corpus. No claim that older training or other corpora have already been checked. Human-supplied title/abstract are not AI-token gold.','source_files_unchanged':True,'training_launched':False}
(OUT/'reservation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['counts'],indent=2));assert len(held)==120
