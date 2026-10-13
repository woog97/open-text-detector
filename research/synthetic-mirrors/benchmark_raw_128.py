"""Four sequential fresh-seed 128-output throughput trials; no quality filtering."""
import asyncio,collections,fcntl,hashlib,json,os,random,statistics,time
from pathlib import Path
import httpx
from run_raw_fast import Production,Ledger,Capacity,save,stamp,GENERATOR
BASE=Path('/tmp/pangram-luna-active-20261002'); ROOT=BASE/'benchmark-128-20261002'; RUN=BASE/'run'
class Batch(Production):
 async def initialize(self):
  self.baseline=0
  for row in self.rows.values():self.inputs.put_nowait(row)
 async def monitor(self):
  while not self.finished:
   await asyncio.to_thread(save,self.root/'raw-production-status.json',self.status('running'))
   if (RUN/'stop-requested.json').exists():self.stop=True
   await asyncio.sleep(1)
async def main():
 ROOT.mkdir(exist_ok=False)
 previous=json.loads((RUN/'raw-production-status.json').read_text());assert previous['state']=='target_reached' and previous['saved_raw_documents']==10000
 manifest=json.loads((RUN/'manifest.json').read_text())
 async with httpx.AsyncClient(headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},timeout=httpx.Timeout(900,connect=20),limits=httpx.Limits(max_connections=256,max_keepalive_connections=256)) as client:
  endpoint=await client.get('https://openrouter.ai/api/v1/models/'+GENERATOR['model']+'/endpoints');endpoint.raise_for_status();flex=next(e for e in endpoint.json()['data']['endpoints'] if e['tag']=='openai/flex')
  for k,v in [('prompt',.05),('completion',.25),('input_cache_write',.0625)]:assert float(flex['pricing'].get(k,0))*1e6<=v+1e-10
  save(ROOT/'endpoint.json',flex)
  async def fetch(body):
   r=await client.post('https://openrouter.ai/api/v1/chat/completions',json=body)
   try:data=r.json()
   except ValueError:data={'error':{'type':'non_json_response'}}
   return r.status_code,data,dict(r.headers)
  ledger=Ledger(RUN,20,manifest['canonical_model'],Capacity(64,64),fetch);start_cost=ledger.spent;start_reserve=ledger.uncertain
  ledger.budget=min(20,start_cost+start_reserve+2)
  used={c['source_record_id'] for c in ledger.calls.values()}
  rows=[json.loads(l) for l in (BASE/'raw-production-input.jsonl').read_text().split('\n') if l.strip()]
  pools=collections.defaultdict(list)
  for r in rows:
   if r['record_id'] not in used and not r.get('generation_seed_excluded'):pools[r['source_id']].append(r)
  # Source/length matched quartets, randomized assignment, distinct seeds in every arm.
  rng=random.Random(12820261002);arms=[[] for _ in range(4)]
  for source,rs in sorted(pools.items()):
   rs.sort(key=lambda r:(len(r['text'].split()),r['record_id']))
   for j in range(0,len(rs)-3,4):
    q=rs[j:j+4];rng.shuffle(q)
    for i in range(4):arms[i].append(q[i])
  # Same quartet order across arms, so initial samples preserve source and length matching.
  order=list(range(len(arms[0])));rng.shuffle(order);arms=[[a[j] for j in order[:192]] for a in arms];assert all(len(a)==192 for a in arms)
  save(ROOT/'design.json',{'batch_size':128,'trials':4,'max_new_documents':512,'incremental_budget_usd':2,'cumulative_budget_usd':20,'baseline_cost':start_cost,'baseline_reserve':start_reserve,'settings':[[64,8],[64,24],[128,32],'repeat_fastest'],'matching':'Fresh unattempted seeds, within-source nearest-length quartets, randomized arm assignment and common quartet order','filtering_started':False,'created_at':stamp()})
  results=[];settings=[(64,8),(64,24),(128,32)]
  for i,arm in enumerate(arms):
   if i==3:
    best=max(results,key=lambda r:r['documents_per_minute']);settings.append((best['concurrency'],best['topic_workers']))
   concurrency,topics=settings[i];out=ROOT/f'batch-{i+1}';out.mkdir();save(out/'seed-ids.json',[r['record_id'] for r in arm])
   ledger.capacity=Capacity(concurrency,concurrency);ledger.timings=collections.defaultdict(list);before_ids=set(ledger.calls);before_cost=ledger.spent;before_reserve=ledger.uncertain;before_requests=ledger.new_requests
   prod=Batch(out,arm,ledger,target=128,topic_workers=topics,writer_workers=concurrency)
   save(ROOT/'status.json',{'state':'running','batch':i+1,'concurrency':concurrency,'topic_workers':topics,'results':results,'filtering_started':False})
   began=time.monotonic();state=await prod.run();elapsed=time.monotonic()-began
   docs=[json.loads(p.read_text()) for p in (out/'raw-documents').glob('*.json')];newcalls=[v for k,v in ledger.calls.items() if k not in before_ids];writers=[c for c in newcalls if c['stage']=='writer'];tokens=sum((c.get('response',{}).get('usage')or{}).get('completion_tokens',0) for c in writers)
   lengths=[len(d['text'].split()) for d in docs]
   result={'batch':i+1,'concurrency':concurrency,'topic_workers':topics,'documents':len(docs),'elapsed_seconds':elapsed,'documents_per_minute':len(docs)*60/elapsed,'documents_per_hour':len(docs)*3600/elapsed,'writer_completion_tokens':tokens,'writer_completion_tokens_per_second':tokens/elapsed,'mean_output_words':statistics.mean(lengths) if lengths else 0,'mean_seed_words':statistics.mean(len(prod.rows[d['source_record_id']]['text'].split()) for d in docs) if docs else 0,'cost_usd':ledger.spent-before_cost,'uncertain_reserve_usd':ledger.uncertain-before_reserve,'requests':ledger.new_requests-before_requests,'errors':prod.errors,'backoffs':ledger.capacity.backoffs,'final_concurrency':ledger.capacity.limit,'stage_timings':state['stage_timings'],'state':state['state'],'filtering_started':False}
   results.append(result);save(out/'result.json',result);print(json.dumps(result),flush=True)
   save(ROOT/'status.json',{'state':'between_batches','results':results,'cumulative_cost_usd':ledger.spent,'filtering_started':False})
   if state['state']!='target_reached' or ledger.halted:break
  save(ROOT/'results.json',{'state':'complete' if len(results)==4 and all(r['documents']==128 for r in results) else 'incomplete','results':results,'new_documents':sum(r['documents'] for r in results),'cost_usd':ledger.spent-start_cost,'cumulative_cost_usd':ledger.spent,'uncertain_reserve_usd':ledger.uncertain,'filtering_started':False,'finished_at':stamp()})
  save(ROOT/'status.json',json.loads((ROOT/'results.json').read_text()))
if __name__=='__main__':
 with (RUN/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main())
