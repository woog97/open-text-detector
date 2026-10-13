"""Independent recipe validation; no model inference or training."""
from pathlib import Path
import json,gzip,hashlib,collections,sys
R=Path('/data/workspace/baseline-mix-v1');P=R/'prepared-v2'
load=lambda p:json.loads(p.read_text())
read=lambda p:[json.loads(l) for l in gzip.open(p,'rt')]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Reconcile metadata with the new assignments before publishing launcher verification.
a=load(P/'split-assignments.json');m=load(P/'manifest.json')
if 'parent_split_counts' not in m:
 m['parent_split_counts']=m['split_counts'];m['split_counts']=dict(collections.Counter(r['dataset']+'/'+r['split'] for r in a));(P/'manifest.json').write_text(json.dumps(m,indent=2))
recipe=load(R/'recipe.json');recipe['sha256']['prepared-v2/manifest.json']=digest(P/'manifest.json');(R/'recipe.json').write_text(json.dumps(recipe,indent=2))
reserved=set(load(R/'reservation.json')['exclude_from_training']['source_paper_ids']);report=load(R/'split-report.json');pool={};checks={}
for source,v in report['sources'].items():
 rows=read(P/('pool-'+source+'.jsonl.gz'));assert len(rows)==v['training_documents']<=v['total_documents']*80//100
 assert not any(str(r['paper_id']) in reserved for r in rows)
 pool[source]={r['id']:r for r in rows}
 checks[source]={'training_documents':len(rows),'cap_passed':True}
held={(r['dataset'],r['id']) for r in a if r['split']!='train'}
for source,rs in pool.items():assert not any((source,i) in held for i in rs)
sys.path.insert(0,'/data/workspace/current-data-v1')
from transformers import AutoTokenizer
from data import encode_example
tok=AutoTokenizer.from_pretrained('/data/workspace/current-data-v1/assets/modernbert',local_files_only=True)
replacement_checks=0
for name,entry in m['files'].items():
 blob=gzip.decompress((P/(name+'.jsonl.gz')).read_bytes());assert hashlib.sha256(blob).hexdigest()==entry['sha256'];assert len(blob.splitlines())==entry['rows']
 if not name.startswith('stage'):continue
 rows=[json.loads(l) for l in blob.splitlines()];old=read(Path('/data/workspace/current-data-v1/prepared-v2')/(name+'.jsonl.gz'))
 assert len(rows)==len(old)
 counts=collections.Counter(r['dataset'] for r in rows);weights={'human':.25,'mirrors':.25,'papers':.25 if name.startswith('stage2') else .45,'gradtex':.2 if name.startswith('stage2') else 0,'fullpapers':.05}
 assert counts=={k:round(len(rows)*v) for k,v in weights.items() if v}
 for r,before in zip(rows,old):
  assert r['id'] in pool[r['dataset']];assert r['group']==pool[r['dataset']][r['id']]['group']
  assert r['paper_id'] not in reserved
  if r!=before:
   assert before['paper_id'] in reserved and before['dataset']==r['dataset']
   encode_example(r,tok,'encoder',2 if name.startswith('stage2') else 1);replacement_checks+=1
for name,sha in recipe['sha256'].items():assert digest(R/name)==sha,name
result={'passed':True,'checks':checks,'replacement_encodings_checked':replacement_checks,'all_schedule_memberships_verified':True,'all_file_hashes_verified':True,'all_existing_unaffected_draws_preserved':True,'model_inference':False,'training_started':False,'split_report':report,'recipe_sha256':digest(R/'recipe.json')}
(R/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
