"""Deterministic export and accounting; never runs a detector or modifies generated prose."""
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import paper_workflow_eval as w
b,OUT=w.b,w.OUT

def main():
 papers={p['paper_id']:p for p in w.rows('papers.jsonl')};ps=w.rows('passages.jsonl')
 jobs={j['id']:j for p in ps for j in w.build_jobs(p)}
 responses={r['request_id']:r for r in w.rows('responses.jsonl')}
 raw=w.rows('outputs.jsonl');outputs={r['id']:r for r in raw}
 assert len(raw)==len(outputs),'Duplicate completed output IDs'
 dataset=[];controls=[]
 for p in ps:
  meta=papers[p['paper_id']]
  for view,text in [('paragraph',p['held_out']),('context',p['before']+'\n\n'+p['held_out']+'\n\n'+p['after'])]:
   controls.append({'id':p['passage_id']+'/human/'+view,'family_id':p['passage_id'],'paper_id':p['paper_id'],'forum_id':p.get('forum_id'),'split':p['split'],'view':view,'condition':'untouched','text':text,'regions':[{'start':0,'end':len(text),'label':'human'}],'sentences':w.source.sentence_spans(text),'label_policy':'historical_human_provenance','pdf_sha256':meta['pdf_sha256'],'pdf_url':meta['pdf_url'],'human_label_basis':meta['human_label_basis']})
 for rid,out in outputs.items():
  j=jobs[rid];meta=papers[j['paper_id']]
  assert responses[rid+'/writer']['output']['paragraph']==out['candidate']
  w.parse_judge(json.dumps(out['quality']),j,out['candidate'])
  for r in w.exported(j,out['candidate']):
   r.update(quality=out['quality'],quality_stratum=out['quality_stratum'],requested_sentence_count=1 if j['condition']=='sentence' else 2 if j['condition']=='two_sentence' else None,generated_sentence_count=len(w.source.sentence_spans(out['candidate'])),generation_id=responses[rid+'/writer']['generation_id'],judge_generation_id=responses[rid+'/judge']['generation_id'],quality_review_origin=responses[rid+'/judge'].get('quality_review_origin','Luna'),quality_evidence_corrections=responses[rid+'/judge'].get('evidence_corrections',[]),conference=meta['conference'],year=meta['year'],pdf_url=meta['pdf_url'],pdf_sha256=meta['pdf_sha256'],development_exposed=j['split']=='pilot')
   dataset.append(r)
 def write(name,xs):
  tmp=OUT/(name+'.tmp');b.writel(tmp,xs);tmp.replace(OUT/name)
 write('dataset.jsonl',dataset);write('matched-human-controls.jsonl',controls)
 for split in ['pilot','calibration','test']:
  write(split+'-reconstruction.jsonl',[r for r in dataset if r['split']==split and r['label_policy']=='known_replacement_provenance'])
  write(split+'-assistance.jsonl',[r for r in dataset if r['split']==split and r['label_policy']=='human_origin_ai_assisted_no_binary_gold'])
  write(split+'-matched-human.jsonl',[r for r in controls if r['split']==split])
  write(split+'-human-novel-clean.jsonl',[r for r in w.rows('human-body-audited.jsonl') if r['split']==split and r['eligible_clean_novel_control']])
 costs=w.accounting()
 if (OUT/'key-after.json').exists():
  before=json.loads((OUT/'key-before.json').read_text())['usage'];after=json.loads((OUT/'key-after.json').read_text())['usage']
  billed=sum(Decimal(str(a['response'].get('usage',{}).get('cost',0))).quantize(Decimal('0.000000001'),rounding=ROUND_DOWN) for a in w.rows('pilot-v1/attempts.jsonl')+w.rows('pilot-v2/attempts.jsonl')+w.rows('attempts.jsonl'))
  costs.update(account_usage_delta_usd=after-before,ledger_account_precision_usd=float(billed),account_matches_ledger=abs(after-before-float(billed))<1e-7)
 w.save('costs.json',costs)
 summary={'at_utc':b.now(),'papers':len(papers),'planned_papers':189,'generated_targets':len(outputs),'planned_targets':189*7,'correlated_views':len(dataset),'matched_human_views':len(controls),'human_body_paragraphs':len(w.rows('human-body.jsonl')),'clean_novel_human_paragraphs':sum(r['eligible_clean_novel_control'] for r in w.rows('human-body-audited.jsonl')),'by_split':dict(Counter(r['split'] for r in outputs.values())),'by_condition':{c:{'count':sum(r['condition']==c for r in outputs.values()),'fidelity':dict(Counter(r['quality']['fidelity'] for r in outputs.values() if r['condition']==c)),'faithful_non_degraded':sum(r['quality_stratum'] for r in outputs.values() if r['condition']==c),'requested_sentence_count_mismatches':sum(r['generated_sentence_count']!=(1 if c=='sentence' else 2) for r in outputs.values() if r['condition']==c) if c in ['sentence','two_sentence'] else None} for c in w.CONDITIONS},'review_repairs':w.rows('review-repairs.jsonl'),'missing_ids':sorted(set(jobs)-set(outputs)),'costs':costs,'complete':len(papers)==189 and len(outputs)==189*7,'limitations':['Luna-only workflow transfer; no unseen-generator claim','Automated Luna quality ratings are not independent expert gold','Assistance edits have no binary token-authorship gold','Historical prose and author-name filtering produce a selected source population','Two views and seven conditions per paper are correlated; bootstrap by paper']}
 w.save('summary.json',summary)
 report=['# Evaluation generation results','',f"{len(papers)} papers; {len(outputs)} generated targets; {len(dataset)} correlated views.",'', '| Condition | Outputs | Equivalent | Minor | Material | Uncertain |','|---|---:|---:|---:|---:|---:|']
 for c,d in summary['by_condition'].items():
  q=d['fidelity'];report.append(f"| {c} | {d['count']} | {q.get('equivalent',0)} | {q.get('minor_difference',0)} | {q.get('material_difference',0)} | {q.get('uncertain',0)} |")
 report+=['','Quality judgments are from Luna, not independent expert annotation. All valid outputs are retained. No classifier performance has been calculated on this collection.','', '| API stage, including archived pilots | Input tokens | Output tokens | Cost USD |','|---|---:|---:|---:|']
 for stage,c in costs['by_stage'].items():report.append(f"| {stage} | {c['input_tokens']:,} | {c['output_tokens']:,} | ${c['cost_usd']:.6f} |")
 report+=['',f"Total API charges: **${costs['cost_usd']:.6f}**, including ${costs['archived_pilots_cost_usd']:.6f} from archived pilots.",'',f"Untouched human body pool: {summary['human_body_paragraphs']:,} paragraphs; {summary['clean_novel_human_paragraphs']:,} clean, novel controls across all splits.",'', 'See scoring/manifest.json for prepared suite hashes, frozen thresholds, and the separate ELLIPSE supplement.']
 (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
 (OUT/'README.md').write_text('''# Paper workflow evaluation\n\nEvaluation-only Luna Flex corpus. See PROTOCOL.md for sampling, two design reviews, metrics and limitations; summary.json for coverage, quality and costs.\n\nUse test-reconstruction.jsonl plus matched human controls for provenance-based token/sentence detection. `regions` are half-open character intervals into `text`; only `text` is model input. Select one view per primary analysis or report views separately. Keep all rows sharing paper_id together and bootstrap by paper. `forum_id` is null unless independently verified.\n\nUse calibration only for explicitly labeled threshold recalibration; primary comparisons retain the prior frozen thresholds. Never train on pilot, calibration or test in this evaluation release. Pilot is development-exposed.\n\nAssistance rows are separate diagnostics: changed text is not binary authorship truth. Unchanged runs and diff operations support overflagging/edit-extent analysis. Compare all mechanically valid outputs first; faithful_non_degraded is a secondary, automated quality stratum.\n\nUse human-novel-clean files for the clean remaining-paragraph audit, and human-body-audited.jsonl for extraction/overlap slices. Historical provenance supports human labels but is not observed authorship. Sentence boundaries are heuristic. Requested and actual sentence counts are retained; report count mismatches separately when interpreting one/two-sentence conditions. See OPERATIONS.md for the recorded evidence-quote repairs; no semantic verdict was manually changed. All model inference must use reduced precision.\n\nRequest/response journals contain prompts, model/tier IDs, usage and provider-reported charges. Same-model quality judgments do not establish human-quality gold. This collection tests Luna workflows, not unseen model families.\n''')
 print(json.dumps({k:summary[k] for k in ['papers','generated_targets','complete','costs']},indent=2))
if __name__=='__main__':
 main()
 import prepare_workflow_suite
 prepare_workflow_suite.main()
