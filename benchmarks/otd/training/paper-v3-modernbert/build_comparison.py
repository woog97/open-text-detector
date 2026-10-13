"""Deterministic coverage subset; selection never reads detector predictions."""
import json,gzip,hashlib,collections
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).parent;OUT=ROOT/'comparison-v1';SEED='comparison-v1-20261001'
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def key(r):return sha(SEED+r['id'])
def read(p):return [json.loads(l) for l in p.open()]
def sample(rs,fields,n):
 groups=collections.defaultdict(list)
 for r in rs:groups[tuple(str(r.get(f,'')) for f in fields)].append(r)
 return [r for v in groups.values() for r in sorted(v,key=key)[:n]]
def save_rows(name,rows):
 blob=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(OUT/(name+'.jsonl.gz')).write_bytes(gzip.compress(blob,mtime=0));return hashlib.sha256(blob).hexdigest()
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'manifest.json').exists()
 rows=read(ROOT/'wide-eval-v1/suite.jsonl');by=collections.defaultdict(list)
 conflicts={i for g in json.loads((ROOT/'wide-eval-v1/span_annotation_conflicts.json').read_text())['conflicts'] for i in g['ids']}
 for r in rows:
  r['span_annotation_conflict']=r['id'] in conflicts;by[r['dataset']].append(r)
 chosen=[]
 for name,rs in by.items():
  if name=='human_paper_remaining':
   for r in rs:r['selection_cohort']='novel_clean' if r['cohort']=='novel_body' and r['clean_prose'] else 'novel_flagged' if r['cohort']=='novel_body' else 'previous_context'
   ss=sample([r for r in rs if r['selection_cohort']=='novel_clean'],['paper_id'],2)+sample([r for r in rs if r['selection_cohort']!='novel_clean'],['paper_id','selection_cohort'],1)
  elif name=='arena50':
   ids=sorted({r['group_id'] for r in rs},key=lambda x:sha(SEED+x))[:20];ss=[r for r in rs if r['group_id'] in ids]
  elif name=='opai':
   groups=sample([{'id':r['group_id'],'domain':r.get('domain')} for r in {x['group_id']:x for x in rs}.values()],['domain'],20);ids={r['id'] for r in groups};ss=[r for r in rs if r['group_id'] in ids]
  elif name in ['meld_eval','detectrl','gede']:ss=sample(rs,['cohort','label','generator','domain','attack','operation'],4)
  elif name=='pelic':ss=sample(rs,['length_bucket'],100)
  else:ss=rs
  # Fill metadata-only coverage gaps, including rare fully-AI OpAI trajectories.
  selected={r['id'] for r in ss}
  for field in ['label','generator','domain','attack','operation','year','conference','length_bucket']:
   available={str(r[field]) for r in rs if r.get(field) is not None};covered={str(r[field]) for r in ss if r.get(field) is not None}
   for value in sorted(available-covered):
    extra=sorted([r for r in rs if str(r.get(field))==value],key=key)[:4]
    if name=='opai':
     groups={r['group_id'] for r in extra};extra=[r for r in rs if r['group_id'] in groups]
    for r in extra:
     if r['id'] not in selected:ss.append(r);selected.add(r['id'])
  chosen+=ss
 ids={r['id'] for r in chosen if r['dataset']=='human_paper_remaining'}
 for r in read(ROOT/'wide-eval-context-v1/suite.jsonl'):
  if r['paired_isolated_id'] in ids:r['dataset']='human_paper_context';chosen.append(r)
 export=ROOT.parents[1]/'exports/ai-paper-provenance-v3-10000/data/fresh_passages'
 testids=set(json.loads((ROOT/'run-01/test_subset_ids.json').read_text()));val_ids=set(json.loads((ROOT/'run-01/data_manifest.json').read_text())['validation_ids'])
 val=[]
 for split,allowed in [('test',testids),('validation',val_ids)]:
  for r in pq.read_table(export/f'{split}-00000-of-00001.parquet').to_pylist():
   if r['id'] not in allowed:continue
   r.update(dataset='paper_v3_target',group_id=r['paper_id'],text_sha256=sha(r['text']),label='human' if r['target_label']==0 else 'mixed',granularity='observed_edit_spans',cohort=r['variant'])
   if split=='test':
    r['regions']=[{'start':0,'end':r['target_start'],'label':-100},{'start':r['target_start'],'end':r['target_end'],'label':r['target_label']},{'start':r['target_end'],'end':len(r['text']),'label':-100}];chosen.append(r)
   else:val.append(r)
 assert len(val)==1048 and len([r for r in chosen if r['dataset']=='paper_v3_target'])==1038
 assert not {r['paper_id'] for r in val}&{r.get('paper_id') for r in chosen}
 assert len({r['id'] for r in chosen})==len(chosen)
 manifest={'seed':SEED,'selection':'Score-independent SHA ranking. All small suites. MELD/DetectRL/GEDE: 4 per native stratum; PELIC:100 per length; Arena:20 shared prompts; OpAI:20 source groups per domain with all trajectories; each historical test paper:2 clean novel,1 flagged novel,1 previous-context paragraphs where available, paired with neighbor context. All 1038 quality-filtered v3 target test rows. Fill any missing categorical metadata level with up to4 examples (whole source groups for OpAI); no selection by performance.','selection_reads_predictions':False,'test_rows':len(chosen),'validation_rows':len(val),'datasets':dict(collections.Counter(r['dataset'] for r in chosen)),'files':{'suite':save_rows('suite',chosen),'validation':save_rows('validation',val)},'calibration':'Same 1048 original held-out validation inputs, token and sentence thresholds <=1% human FPR; document mean reference-token score threshold <=1% on human-original validation rows. No test calibration.','reference_grid':'Our ModernBERT tokenizer offsets; each model token score projected by character-overlap-weighted mean; sentences average reference-token scores; special/whitespace tokens excluded. Reference tokens crossing provenance boundaries ignored.','runtime':'Our model: original BF16 510-content overlapping windows stride256; MELD: BF16 nonoverlapping2046-content windows, no input truncation. Full length may exceed publisher capped scoring contract.','caveat':'Diagnostic suite already inspected; public benchmark training overlap with MELD is unknown. Unequal sampling fractions; report per-dataset, never pooled accuracy as population estimate.'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
