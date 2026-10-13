import collections,json,math
from pathlib import Path
P=Path(__file__).resolve().parent;rows=json.loads((P/'dataset.json').read_text());protocol=json.loads((P/'protocol.json').read_text());reports={}
for name in protocol['prompts']:
 d=P/name;r=json.loads((d/'results.json').read_text());assert r['state']=='complete' and r['completed']==216 and not r['errors'];assert r['reported_tiers']=={'flex':216}
 preds=json.loads((d/'predictions.json').read_text());index={x['id']:x for x in preds};assert len(index)==216
 cost=0;counts=collections.Counter();tokens=collections.Counter()
 for row in rows:
  import hashlib
  call=json.loads((d/'calls'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text());resp=call['response'];request=call['request']
  assert request['messages']==[{'role':'system','content':protocol['prompts'][name]},{'role':'user','content':row['text']}]
  predicted=json.loads(resp['choices'][0]['message']['content'])['label'];assert predicted==index[row['id']]['prediction']
  assert resp['service_tier']=='flex' and resp['model']=='openai/gpt-6-luna'
  cost+=resp['usage']['cost'];counts[row['profile'],row['label'],predicted]+=1
  for k in ['prompt_tokens','completion_tokens']:tokens[k]+=resp['usage'][k]
 assert math.isclose(cost,r['cost_usd'])
 for profile in ['comparison','reconstruction']:
  for k,pair in [('tn',('HUMAN','HUMAN')),('fp',('HUMAN','AI')),('tp',('AI','AI')),('fn',('AI','HUMAN'))]:assert counts[(profile,*pair)]==r['profiles'][profile][k]
 reports[name]=r|{'verified_raw_responses':True,'tokens':dict(tokens)}
(P/'comparison.json').write_text(json.dumps(reports,indent=2)+'\n')
lines=['Small broader Luna evaluation — October 3, 2026','',
'216 saved inputs, two frozen prompts, 432 calls. All responses confirmed Luna and Flex. The prompts were not changed after viewing these results.',
'Comparison: 120 passages, 40 human/40 AI/40 mixed across 19 source datasets, sampled round-robin by dataset and once per source group. This is a diversity sample, not a population-weighted estimate.',
'Reconstruction: 12 previously unused calibration papers, with one untouched control and four generated conditions each (60 passages).',
'Assistance: proofreading and light-polish outputs from the same 12 calibration papers (24 passages); AI flag rates only, not accuracy under the frozen substantial-rewrite prompt.',
'Manuscripts: 12 generated papers, four per final revision model. Positive-only detection; no matched human controls or accuracy/FPR estimate.',
'Excluded all 108 prompt-tuning paper IDs and five demonstration paper IDs plus their exact texts. The calibration papers retain their historical split; this is not a newly sealed test set or a comprehensive cross-corpus near-duplicate/pretraining-overlap audit. Samples and original labels are in the four *-mini.jsonl files.', '']
for name,r in reports.items():
 lines.append(name)
 lines.append(f"API cost ${r['cost_usd']:.8f}")
 for profile,pr in r['profiles'].items():
  lines.append(profile+':')
  if 'balanced_accuracy' in pr:lines.append(f"  Balanced accuracy {pr['balanced_accuracy']:.2%}; human false positives {pr['fp']}/{pr['human_n']}; AI-involved recall {pr['tp']}/{pr['ai_involved_n']}.")
  for label,s in pr['slices'].items():lines.append(f"  {label}: flagged AI {s['flagged_ai']}/{s['n']} ({s['ai_flag_rate']:.2%}).")
 lines.append('')
lines+=['Total API cost $'+format(sum(r['cost_usd'] for r in reports.values()),'.8f'),
'Verified input/prompt payloads, model identity, reported Flex tier, raw-response labels, confusion counts, and billing totals. No model assets downloaded, source datasets modified, or fresh text generation performed. Hosted inference precision is provider-managed and unreported. Tiny sample counts make per-slice estimates noisy.']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
