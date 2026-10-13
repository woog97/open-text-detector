"""Coverage and fixed-threshold sensitivity audit after score-independent selection."""
import json,gzip,collections
from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'comparison-v1'
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else p.open()) as f:return [json.loads(l) for l in f]
full=read(ROOT/'wide-eval-v1/suite.jsonl');sub=read(OUT/'suite.jsonl.gz');pred={r['id']:r for r in read(ROOT/'wide-eval-v1/predictions.jsonl.gz')};report={}
for ds in sorted({r['dataset'] for r in full}):
 a=[r for r in full if r['dataset']==ds];b=[r for r in sub if r['dataset']==ds];result={'full_rows':len(a),'selected_rows':len(b),'coverage':{}}
 for field in ['label','generator','domain','attack','operation','year','conference','length_bucket']:
  allvals={str(r[field]) for r in a if r.get(field) is not None};newvals={str(r[field]) for r in b if r.get(field) is not None}
  if allvals:result['coverage'][field]={'available':len(allvals),'selected':len(newvals),'missing':sorted(allvals-newvals)}
 for condition in ['all','novel_clean'] if ds=='human_paper_remaining' else ['all']:
  stats={}
  for name,rr in [('full',a),('subset',b)]:
   if condition=='novel_clean':rr=[r for r in rr if r['cohort']=='novel_body' and r['clean_prose']]
   unique={r['text_sha256']:pred[r['id']] for r in rr};cs=[r['token_counts'] for r in unique.values() if r['token_counts'] is not None]
   if cs:
    tp,fp,fn,tn=map(sum,zip(*cs));stats[name]={'token_recall':tp/(tp+fn) if tp+fn else None,'human_fpr':fp/(fp+tn) if fp+tn else None}
  result[condition]=stats
 report[ds]=result
(OUT/'coverage.json').write_text(json.dumps({'performance_used_to_select_examples':False,'metadata_coverage_completion_rows':12,'note':'Post-selection diagnostic only; original frozen thresholds. These raw counts do not yet exclude known annotation conflicts; final comparison does. Selection preserved regardless of performance differences.','datasets':report},indent=2))
for ds,d in report.items():print(ds,d['selected_rows'],[(f,v['missing']) for f,v in d['coverage'].items() if v['missing']],d.get('novel_clean',d['all']))
