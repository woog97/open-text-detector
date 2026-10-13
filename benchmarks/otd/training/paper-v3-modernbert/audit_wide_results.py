"""Audit completed predictions and exclude score-independent annotation conflicts."""
from pathlib import Path
import json,collections,hashlib,time
from score_wide_eval import aggregate,write,progress

def main(out):
 out=Path(out);manifest=json.loads((out/'manifest.json').read_text());audit=json.loads((out/'span_annotation_conflicts.json').read_text());excluded={i for c in audit['conflicts'] for i in c['ids']}
 original=json.loads((out/'results.json').read_text())
 if not (out/'results_before_annotation_audit.json').exists():write(out/'results_before_annotation_audit.json',original)
 predictions=[json.loads(l) for l in (out/'predictions.jsonl').open()]
 assert len(predictions)==len({r['id'] for r in predictions})==manifest['evaluation_rows']
 assert len({r['text_sha256'] for r in predictions})==manifest['unique_evaluation_texts']
 for r in predictions:r['span_annotation_conflict']=r['id'] in excluded
 result={k:v for k,v in original.items() if k not in ['datasets','breakdowns']};result['datasets']={};result['breakdowns']={};result['annotation_audit']={'conflicting_exact_texts':len(audit['conflicts']),'excluded_span_metric_rows':len(excluded),'policy':audit['policy'],'score_independent':True}
 progress(out,'auditing_annotations',conflicting_texts=len(audit['conflicts']))
 for dataset in sorted({r['dataset'] for r in predictions}):
  rr=[r for r in predictions if r['dataset']==dataset];result['datasets'][dataset]=aggregate(rr)
  fields=['cohort','generator','domain','attack','length_bucket']
  if dataset=='human_paper_remaining':fields+=['conference','year','clean_prose']
  result['breakdowns'][dataset]={}
  for field in fields:
   parts=collections.defaultdict(list)
   for r in rr:
    if r.get(field) is not None:parts[str(r[field])].append(r)
   if len(parts)>1:result['breakdowns'][dataset][field]={k:aggregate(v,False) for k,v in sorted(parts.items())}
  if dataset=='human_paper_remaining':
   for name,rs in [('novel_clean',[r for r in rr if r['cohort']=='novel_body' and r['clean_prose']]),('novel_all',[r for r in rr if r['cohort']=='novel_body']),('previous_context',[r for r in rr if r['cohort']=='previous_context'])]:result['breakdowns'][dataset][name]=aggregate(rs)
 write(out/'results.json',result)
 verified={'prediction_rows':len(predictions),'all_unique_ids':True,'suite_count_match':True,'model_sha256':manifest['checkpoint_sha256'],'frozen_thresholds':manifest['thresholds'],'new_generation_calls':0,'annotation_conflicts_excluded_from_span_metrics':len(excluded),'results_sha256':hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest(),'verified_at':time.time()}
 write(out/'verification.json',verified);progress(out,'complete',rows=len(predictions),annotation_audit_complete=True)
 print(json.dumps(verified),flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1])
