"""Derive immutable baseline recipe without modifying an existing run."""
from pathlib import Path
import json,gzip,hashlib,shutil,random,sys,collections,time
from policy import TOTALS,STAGE2,blocked_groups,cap_groups,validate
BASE=Path('/data/workspace/current-data-v1');ROOT=Path('/data/workspace/baseline-mix-v1');P=ROOT/'prepared-v2'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def read(p):return [json.loads(l) for l in gzip.open(p,'rt')]
def write(name,rows):
 b=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(P/(name+'.jsonl.gz')).write_bytes(gzip.compress(b,mtime=0));return {'rows':len(rows),'papers':len({r['paper_id'] for r in rows}),'sha256':hashlib.sha256(b).hexdigest()}
def main():
 assert not (ROOT/'recipe.json').exists(),'Already built; do not overwrite'
 P.mkdir(exist_ok=False);(ROOT/'configs').mkdir();(ROOT/'trainer').mkdir()
 old=BASE/'prepared-v2';m=load(old/'manifest.json')
 pools={s:read(old/('pool-'+s+'.jsonl.gz')) for s in TOTALS}
 for s,rs in pools.items():assert len(rs)==m['files']['pool-'+s]['rows']
 reservation=load(ROOT/'reservation.json');reserved=set(reservation['exclude_from_training']['source_paper_ids'])
 reserved_blocked=blocked_groups(pools,reserved);blocked=cap_groups(pools,reserved_blocked)
 kept={s:[r for r in rs if r['group'] not in blocked] for s,rs in pools.items()}
 removed={s:[r for r in rs if r['group'] in blocked] for s,rs in pools.items()}
 report={'created_at':time.time(),'unit':'source records/documents; not sampled repetitions or token windows','sources':{},'reserved_source_families':len(reserved),'removed_connected_groups':len(blocked),'historical_schedule_overlap':{}}
 for s,rs in pools.items():
  report['sources'][s]={'total_documents':TOTALS[s],'previous_training_documents':len(rs),'training_documents':len(kept[s]),'training_percent':100*len(kept[s])/TOTALS[s],'outside_training':TOTALS[s]-len(kept[s]),'removed_for_reservation_or_cap':len(removed[s]),'cap':TOTALS[s]*80//100}
 # Keep all previous validation/test/exclusion assignments; attach the explicit reservation.
 assignments=load(old/'split-assignments.json')
 for r in assignments:
  if r['group'] in blocked and r['split']=='train':r['split']='reserved_evaluation';r['reason']='explicit_manuscript_family_holdout' if r['group'] in reserved_blocked else '80_percent_cap'
 (P/'split-assignments.json').write_text(json.dumps(assignments))
 files={}
 for s,rs in kept.items():files['pool-'+s]=write('pool-'+s,rs)
 for s,rs in removed.items():
  if rs:files['reserved-'+s]=write('reserved-'+s,rs)
 sys.path.insert(0,str(BASE));from data import crop
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(BASE/'assets/modernbert',local_files_only=True)
 def window(r,rng):
  if r.get('supervision')=='document_only':return dict(r)
  off=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];n=len(off);start=rng.randrange(max(1,n-510+1));end=min(n,start+510)
  row=dict(r);row.setdefault('kind','paired' if r['dataset']=='papers' else 'novel_human' if r.get('label')==0 else 'generated')
  if 'regions' not in row:row.update(regions=[{'start':0,'end':len(r['text']),'label':r['label']}],target_start=0,target_end=len(r['text']))
  out=crop(row,off[start][0],off[end-1][1])
  while len(tok(out['text'],add_special_tokens=False)['input_ids'])>510:
   end-=1;out=crop(row,off[start][0],off[end-1][1])
  out.update(dataset=r['dataset'],group=r['group']);return out
 schedules={}
 for name in m['files']:
  if name.startswith('stage'):
   rows=read(old/(name+'.jsonl.gz'));rng=random.Random(name+'/baseline-mix-v1');bad=[r for r in rows if r['group'] in blocked];report['historical_schedule_overlap'][name]={'draws':len(bad),'unique_documents':len({r['id'] for r in bad})}
   for i,r in enumerate(rows):
    if r['group'] in blocked:
     out=window(rng.choice(kept[r['dataset']]),rng);out['draw_id']=r['draw_id'];rows[i]=out
   schedules[name]=rows;files[name]=write(name,rows)
 for name in ['selection-windows','calibration-windows']:
  shutil.copy2(old/(name+'.jsonl.gz'),P/(name+'.jsonl.gz'));files[name]=m['files'][name]
 validate(kept,schedules,blocked)
 assert not ({r['group'] for rs in kept.values() for r in rs}&{r['group'] for r in assignments if r['split'] in ['validation','test','reserved_evaluation']})
 m['parent_split_counts']=m.get('split_counts');m['split_counts']=dict(collections.Counter(r['dataset']+'/'+r['split'] for r in assignments))
 m.update(files=files,pool_counts={s:len(rs) for s,rs in kept.items()},stage2_mix=STAGE2,recipe='baseline-mix-v1',parent_manifest_sha256=digest(old/'manifest.json'),reservation_sha256=digest(ROOT/'reservation.json'),policy='Preserve existing splits; training <=80% of each full collection; explicit manuscript source-family reservation removed from every training pool and schedule. Non-training counts include exclusions; do not treat excluded rows as evaluation gold.')
 (P/'manifest.json').write_text(json.dumps(m,indent=2))
 for p in BASE.glob('*.py'):shutil.copy2(p,ROOT/'trainer'/p.name)
 shutil.copy2(BASE/'configs/encoder.json',ROOT/'configs/encoder.json');shutil.copy2(BASE/'models.lock.json',ROOT/'models.lock.json')
 # Shared dependencies/model assets remain at their existing Space paths.
 report['evaluation_sources']={'human_mirrors':'prepared-v2/split-assignments.json + existing source releases','papers':'original validation/test parquet splits, 4000 rows each','gradtex':'original validation 23672 / test 39477 parquets','manuscripts':'reservation.json and evaluation assignment files','excluded':'not automatically valid evaluation examples'}
 (ROOT/'split-report.json').write_text(json.dumps(report,indent=2))
 paths=[*P.glob('*'),* (ROOT/'trainer').glob('*.py'),ROOT/'configs/encoder.json',ROOT/'models.lock.json',ROOT/'reservation.json',ROOT/'worker.py',ROOT/'track.py']
 recipe={'version':1,'name':'baseline-mix-v1','parent':str(BASE),'mix':STAGE2,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths},'launch':'local spawn.py --name NAME [--gpu INDEX]','model':'ModernBERT-large LoRA; other backbones require their own validated trainer'}
 (ROOT/'recipe.json').write_text(json.dumps(recipe,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
