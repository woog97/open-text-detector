import asyncio,collections,fcntl,json,sys
from pathlib import Path
import engine as p
HERE=Path(__file__).resolve().parent;ROWS=json.loads((HERE/'dataset.json').read_text());PROMPTS=json.loads((HERE.parent/'luna-flex-hillclimb-20261003/prompts.json').read_text());NAMES=['v4_calibration_fewshot','v8_diverse_fewshot'];p.dataset=lambda:ROWS

def summarize(rows):
 records=[];cost=0;tiers=collections.Counter();errors=[]
 for row in rows:
  file=p.HERE/'calls'/(p.sha(row['id'])+'.json')
  if not file.exists():continue
  c=json.loads(file.read_text());r=c.get('response',{});cost+=float(r.get('usage',{}).get('cost') or 0)
  if not r:continue
  tiers[str(r.get('service_tier'))]+=1
  if c.get('state')!='complete':errors.append(row['id']);continue
  assert c['request']['messages']==[{'role':'system','content':p.SYSTEM},{'role':'user','content':row['text']}]
  pred=json.loads(r['choices'][0]['message']['content'])['label'];assert pred==c['prediction']
  records.append({k:v for k,v in row.items() if k!='text'}|{'prediction':pred,'response_id':r['id']})
 groups=collections.defaultdict(list)
 for r in records:groups[r['paper_id']].append(r)
 papers=[]
 for key,rs in groups.items():
  flags=sum(r['prediction']=='AI' for r in rs);papers.append({'paper_id':key,'label':rs[0]['label'],'generator':rs[0]['generator'],'chunks':len(rs),'ai_flags':flags,'ai_fraction':flags/len(rs),'majority_prediction':'AI' if flags/len(rs)>.5 else 'HUMAN'})
 h=[r for r in records if r['label']=='HUMAN'];a=[r for r in records if r['label']=='AI'];tn=sum(r['prediction']=='HUMAN' for r in h);tp=sum(r['prediction']=='AI' for r in a)
 result={'state':'complete' if len(records)==len(rows) else 'partial','completed':len(records),'planned':len(rows),'cost_usd':cost,'reported_tiers':dict(tiers),'errors':errors,'section_counts':{'tn':tn,'fp':len(h)-tn,'tp':tp,'fn':len(a)-tp},'human_section_fpr':(len(h)-tn)/len(h) if h else None,'ai_section_recall':tp/len(a) if a else None,'section_balanced_accuracy':(tn/len(h)+tp/len(a))/2 if h and a else None,'papers':papers,'precision_dtype':'provider_managed_unreported'}
 if result['state']=='complete':
  assert len({r['response_id'] for r in records})==len(rows)
  hs=[r for r in papers if r['label']=='HUMAN'];ais=[r for r in papers if r['label']=='AI']
  result['paper_macro_human_section_fpr']=sum(r['ai_fraction'] for r in hs)/len(hs);result['paper_macro_ai_section_recall']=sum(r['ai_fraction'] for r in ais)/len(ais)
  result['paper_majority_human_correct']=sum(r['majority_prediction']=='HUMAN' for r in hs);result['paper_majority_ai_correct']=sum(r['majority_prediction']=='AI' for r in ais)
  result['human_papers_with_any_false_positive']=sum(r['ai_flags']>0 for r in hs)
 p.save(p.HERE/'results.json',result);p.save(p.HERE/'predictions.json',records);return result
p.summarize=summarize
async def main():
 reserve=0
 for name in NAMES:
  p.SYSTEM=PROMPTS[name];reserve+=sum(p.bound(p.body(r)) for r in ROWS)
 assert reserve<2
 p.save(HERE/'protocol.json',{'prompts':{n:PROMPTS[n] for n in NAMES},'worst_case_usd':reserve,'budget_usd':2,'maximum_calls':len(ROWS)*2,'frozen_prompts':True,'section_independence':'Every section in a separate stateless request; only system prompt and current section text. No section headings, IDs, other sections, or prior outputs sent.','paper_rule':'Strict majority of independently predicted chunks; ties HUMAN. Report any-human-section false positives separately.'})
 if '--check' in sys.argv:print(json.dumps({'rows':len(ROWS),'calls':len(ROWS)*2,'worst_case_usd':reserve}));return
 for name in NAMES:
  p.SYSTEM=PROMPTS[name];p.HERE=HERE/name;p.HERE.mkdir(exist_ok=True)
  with (p.HERE/'worker.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);await p.run()
  assert json.loads((p.HERE/'results.json').read_text())['state']=='complete'
if __name__=='__main__':
 with (HERE/'suite.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
