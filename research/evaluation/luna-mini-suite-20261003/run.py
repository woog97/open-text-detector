import asyncio,collections,fcntl,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'luna-flex-pilot-20261003';OLD=HERE.parent/'luna-flex-hillclimb-20261003'
spec=importlib.util.spec_from_file_location('pilot',BASE/'run.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
ROWS=json.loads((HERE/'dataset.json').read_text());PROMPTS=json.loads((OLD/'prompts.json').read_text());NAMES=['v4_calibration_fewshot','v8_diverse_fewshot'];p.dataset=lambda:ROWS

def summary(rows):
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
 reports={}
 for profile in ['comparison','reconstruction','assistance','manuscripts']:
  rs=[r for r in records if r['profile']==profile];out={'n':len(rs),'slices':{}}
  if profile in ['comparison','reconstruction']:
   h=[r for r in rs if r['label']=='HUMAN'];a=[r for r in rs if r['label']=='AI'];tn=sum(r['prediction']=='HUMAN' for r in h);tp=sum(r['prediction']=='AI' for r in a)
   out.update(human_n=len(h),ai_involved_n=len(a),tn=tn,fp=len(h)-tn,tp=tp,fn=len(a)-tp,balanced_accuracy=(tn/len(h)+tp/len(a))/2 if h and a else None)
  key={'comparison':'original_label','reconstruction':'condition','assistance':'condition','manuscripts':'generator'}[profile]
  for value in sorted({str(r.get(key)) for r in rs}):
   sub=[r for r in rs if str(r.get(key))==value];flag=sum(r['prediction']=='AI' for r in sub);out['slices'][value]={'n':len(sub),'flagged_ai':flag,'ai_flag_rate':flag/len(sub)}
  reports[profile]=out
 result={'state':'complete' if len(records)==len(rows) else 'partial','completed':len(records),'planned':len(rows),'cost_usd':cost,'reported_tiers':dict(tiers),'errors':errors,'profiles':reports,'precision_dtype':'provider_managed_unreported'}
 if result['state']=='complete':assert len({r['response_id'] for r in records})==216
 p.save(p.HERE/'results.json',result);p.save(p.HERE/'predictions.json',records);return result
p.summarize=summary
async def main():
 reserve=0
 for name in NAMES:
  p.SYSTEM=PROMPTS[name];reserve+=sum(p.bound(p.body(r)) for r in ROWS)
 assert reserve<2
 p.save(HERE/'protocol.json',{'prompts':{n:PROMPTS[n] for n in NAMES},'worst_case_usd':reserve,'budget_usd':2,'maximum_calls':432,'frozen_before_eval':True,'no_prompt_tuning_on_this_sample':True})
 if '--check' in sys.argv:print(json.dumps({'rows':len(ROWS),'calls':432,'worst_case_usd':reserve}));return
 for name in NAMES:
  p.SYSTEM=PROMPTS[name];p.HERE=HERE/name;p.HERE.mkdir(exist_ok=True)
  with (p.HERE/'worker.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);await p.run()
  assert json.loads((p.HERE/'results.json').read_text())['state']=='complete'
if __name__=='__main__':
 with (HERE/'suite.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
