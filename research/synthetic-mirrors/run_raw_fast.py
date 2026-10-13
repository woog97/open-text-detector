"""Raw Luna production: local saves only in the request path; no output filtering."""
import argparse,asyncio,collections,fcntl,hashlib,json,math,os,random,time
from pathlib import Path
import httpx
from mirror_core import topic_messages,writer_messages
from run_openrouter import request,reservation,GENERATOR,canonical,sha

def stamp():
 import datetime
 return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)

def content(call):
 r=call.get('response') or {};choices=r.get('choices')or[]
 v=(choices[0].get('message')or{}).get('content') if choices else None
 return v if call.get('http_status')==200 and isinstance(v,str) and v.strip() else None

def topic_value(call):
 text=content(call)
 if not text:return None
 try:v=json.loads(text).get('topic')
 except (ValueError,AttributeError):return None
 return v if isinstance(v,str) and v.strip() else None

class OperationalStop(Exception):pass
class UncertainCall(Exception):pass
class BudgetStop(Exception):pass

class Capacity:
 def __init__(self,start=8,maximum=64):self.limit=start;self.maximum=maximum;self.active=0;self.condition=asyncio.Condition();self.successes=0;self.backoffs=0;self.last_backoff=0
 async def enter(self):
  async with self.condition:
   await self.condition.wait_for(lambda:self.active<self.limit);self.active+=1
 async def leave(self):
  async with self.condition:self.active-=1;self.condition.notify_all()
 async def set_limit(self,n):
  async with self.condition:self.limit=max(1,min(n,self.maximum));self.condition.notify_all()
 async def backoff(self):
  self.backoffs+=1;self.last_backoff=time.monotonic();await self.set_limit(max(4,self.limit//2))

class Ledger:
 def __init__(self,root,budget,canonical_model,capacity,fetch):
  self.root=Path(root);self.budget=budget;self.model=canonical_model;self.capacity=capacity;self.fetch=fetch;self.calls={};self.spent=0.;self.uncertain=0.;self.inflight=0.;self.lock=asyncio.Condition();self.halted=False;self.new_requests=0;self.timings=collections.defaultdict(list)
  for p in (self.root/'calls').glob('*.json'):
   c=json.loads(p.read_text());self.calls[c['call_id']]=c
   if c['state']=='complete':self.spent+=float(c['cost_usd'])
   elif c['state'] not in ['rejected_before_execution']:
    self.uncertain+=float(c.get('reserved_usd',0))
  if self.spent+self.uncertain>budget:raise BudgetStop('Existing ledger exceeds cumulative budget')
 async def call(self,row,stage,body):
  cid=sha(canonical({'request':body,'parent':row['record_id'],'stage':stage}));path=self.root/'calls'/(cid+'.json')
  if cid in self.calls:
   c=self.calls[cid]
   if c['state']=='complete':return c
   raise UncertainCall('Existing call not replayed:'+cid)
  hold=reservation(body)
  async with self.lock:
   while self.spent+self.uncertain+self.inflight+hold>self.budget and self.inflight>0 and not self.halted:await self.lock.wait()
   if self.halted:raise OperationalStop('Shared ledger halted')
   if self.spent+self.uncertain+self.inflight+hold>self.budget:raise BudgetStop('Cumulative budget exhausted')
   self.inflight+=hold
  reserved=True
  try:
   for attempt in range(4):
    c={'call_id':cid,'source_record_id':row['record_id'],'stage':stage,'request':body,'reserved_usd':hold,'state':'dispatched','created_at':stamp(),'raw_fast_attempt':attempt,'filtering_performed':False}
    await asyncio.to_thread(save,path,c);self.calls[cid]=c
    if self.halted:raise OperationalStop('Dispatch halted')
    await self.capacity.enter();began=time.monotonic()
    try:
     self.new_requests+=1;status,response,headers=await self.fetch(body)
    except Exception as exc:
     status,response,headers=0,{'error':{'type':type(exc).__name__}},{}
    finally:await self.capacity.leave()
    elapsed=time.monotonic()-began;self.timings[stage].append(elapsed)
    c.update(http_status=status,response=response,elapsed_seconds=elapsed,response_received_at=stamp(),state='needs_reconciliation')
    await asyncio.to_thread(save,path,c);self.calls[cid]=c
    err=response.get('error')or{};meta=err.get('metadata')or{} if isinstance(err,dict) else {}
    # Only explicit router rejections before provider execution are retryable.
    pre_reject=status in (402,429) and meta.get('limit_source') in ('openrouter_in_flight_budget','openrouter_credits','openrouter_key_limit')
    if pre_reject:
     c.update(state='rejected_before_execution',cost_usd=0.0)
     await asyncio.to_thread(save,self.root/'raw-rejected-requests'/f'{cid}-{attempt}.json',c)
     await asyncio.to_thread(save,path,c);self.calls[cid]=c
     if meta.get('limit_source')!='openrouter_in_flight_budget' or attempt==3:raise OperationalStop('Router credit limit: '+str(meta.get('limit_source')))
     await self.capacity.backoff()
     try:delay=float(headers.get('retry-after',2**attempt*5))
     except ValueError:delay=2**attempt*5
     await asyncio.sleep(max(1,min(120,delay))+random.random());continue
    cost=(response.get('usage')or{}).get('cost');valid_cost=isinstance(cost,(int,float)) and math.isfinite(cost) and cost>=0
    if status!=200 or err or not valid_cost:
     c.update(state='interrupted_reserved_no_replay',reconciliation_basis='No reliable paid completion/cost; preserve response and reserve; never replay')
     await asyncio.to_thread(save,path,c);self.calls[cid]=c
     async with self.lock:self.uncertain+=hold
     if status in (401,403):self.halted=True
     if status==429:await self.capacity.backoff()
     raise UncertainCall('Uncertain response saved:'+cid)
    c.update(state='complete',cost_usd=float(cost),tier_verification='verified' if response.get('service_tier')=='flex' else 'unreported' if response.get('service_tier') is None else 'unexpected',reconciliation_basis='Original response and billed cost preserved; raw output unfiltered')
    await asyncio.to_thread(save,path,c);self.calls[cid]=c
    async with self.lock:self.spent+=float(cost)
    if float(cost)>hold or response.get('model') not in (GENERATOR['model'],self.model) or response.get('service_tier') not in (None,'flex'):
     self.halted=True;raise OperationalStop('Provider identity/tier/cost exceeded configured route')
    self.capacity.successes+=1;return c
   raise OperationalStop('Retry bound reached')
  except OperationalStop:
   self.halted=True;raise
  finally:
   if reserved:
    async with self.lock:self.inflight-=hold;self.lock.notify_all()

class Production:
 def __init__(self,root,rows,ledger,target=10000,topic_workers=8,writer_workers=64):
  self.root=Path(root);self.out=self.root/'raw-documents';self.out.mkdir(exist_ok=True);self.rows={r['record_id']:r for r in rows};self.ledger=ledger;self.target=target;self.topic_workers=topic_workers;self.writer_workers=writer_workers;self.saved=set();self.source_counts=collections.Counter();self.slots=0;self.ready=asyncio.Queue(maxsize=128);self.inputs=asyncio.Queue();self.errors=[];self.stop=False;self.finished=False;self.started=stamp();self.started_mono=time.monotonic();self.producers_left=topic_workers;self.first_new_save_elapsed=None;self.last_new_save_elapsed=None
 def record(self,row,call,topic=None):
  response=call['response'];choice=response['choices'][0]
  if topic is None:
   try:topic=json.loads(call['request']['messages'][-1]['content']).get('topic')
   except Exception:topic=None
  return {'source_record_id':row['record_id'],'source_passage_sha256':row['passage_sha256'],'source_id':row['source_id'],'category':row['category'],'parent_document_id':row.get('parent_document_id'),'document_family_id':row.get('document_family_id',row.get('provisional_family_id')),'split':'development_pilot','development_exposed':True,'text':content(call),'topic':topic,'writer_call_id':call['call_id'],'model':response.get('model'),'requested_model':GENERATOR['model'],'requested_service_tier':GENERATOR.get('service_tier'),'reported_service_tier':response.get('service_tier'),'finish_reason':choice.get('finish_reason'),'usage':response.get('usage'),'created_at':call.get('created_at'),'saved_at':stamp(),'filtering_status':'not_run_user_deferred','training_eligible':False,'admission_status':'unfiltered_synthetic_candidate','known_process':'independent_ai_generation'}
 async def initialize(self):
  for c in self.ledger.calls.values():
   if c.get('stage')!='writer' or not content(c):continue
   row=self.rows.get(c['source_record_id'])
   if not row:raise OperationalStop('Existing output seed absent from retained input')
   if row['record_id'] in self.saved:raise OperationalStop('Duplicate writer output parent')
   record=self.record(row,c);p=self.out/(row['record_id']+'.json')
   if not p.exists():await asyncio.to_thread(save,p,record)
   self.saved.add(row['record_id']);self.source_counts[row['source_id']]+=1
  if len(self.saved)>self.target:raise OperationalStop('Already above target')
  self.baseline=len(self.saved)
  # Do not replay any previously attempted writer, including ambiguous/empty results.
  attempted_writers={c['source_record_id'] for c in self.ledger.calls.values() if c.get('stage')=='writer'}
  blocked={c['source_record_id'] for c in self.ledger.calls.values() if c.get('state') not in ('complete','rejected_before_execution')}
  available=[r for r in self.rows.values() if r['record_id'] not in attempted_writers|blocked and not r.get('generation_seed_excluded',False)]
  # Deterministic source interleaving; dedicated seed preparation supplies balanced order.
  for row in available:self.inputs.put_nowait(row)
 async def extractor(self):
  try:
   while not self.stop and len(self.saved)<self.target:
    try:row=self.inputs.get_nowait()
    except asyncio.QueueEmpty:break
    try:
     c=await self.ledger.call(row,'topic',request(topic_messages(row),'topic'));topic=topic_value(c)
     if topic is None:
      self.errors.append({'seed':row['record_id'],'stage':'topic','reason':'no_usable_topic_json'});continue
     await self.ready.put((row,topic))
    except (BudgetStop,OperationalStop) as exc:self.stop=True;self.errors.append({'stage':'topic','reason':str(exc)})
    except UncertainCall as exc:self.errors.append({'stage':'topic','reason':str(exc)})
  finally:self.producers_left-=1
 async def writer(self):
  while not self.stop and len(self.saved)<self.target:
   if not self.producers_left and self.ready.empty():return
   try:row,topic=await asyncio.wait_for(self.ready.get(),timeout=1)
   except asyncio.TimeoutError:continue
   while len(self.saved)+self.slots>=self.target and len(self.saved)<self.target and not self.stop:await asyncio.sleep(.05)
   if self.stop or len(self.saved)>=self.target:return
   self.slots+=1
   try:
    call=await self.ledger.call(row,'writer',request(writer_messages(row,topic),'writer',len(row['text'].split())))
    if content(call):
     record=self.record(row,call,topic);await asyncio.to_thread(save,self.out/(row['record_id']+'.json'),record)
     self.saved.add(row['record_id']);self.source_counts[row['source_id']]+=1
     self.last_new_save_elapsed=time.monotonic()-self.started_mono
     if self.first_new_save_elapsed is None:self.first_new_save_elapsed=self.last_new_save_elapsed
    else:self.errors.append({'seed':row['record_id'],'stage':'writer','reason':'empty_api_content_preserved'})
   except (BudgetStop,OperationalStop) as exc:self.stop=True;self.errors.append({'stage':'writer','reason':str(exc)})
   except UncertainCall as exc:self.errors.append({'stage':'writer','reason':str(exc)})
   finally:self.slots-=1
 def status(self,state):
  timings={k:{'n':len(v),'mean_seconds':sum(v)/len(v)} for k,v in self.ledger.timings.items() if v}
  return {'state':state,'target_raw_documents':self.target,'saved_raw_documents':len(self.saved),'baseline_raw_documents':self.baseline,'new_raw_documents':len(self.saved)-self.baseline,'active_writer_slots':self.slots,'ready_topics':self.ready.qsize(),'pending_seeds':self.inputs.qsize(),'concurrency_limit':self.ledger.capacity.limit,'active_api_requests':self.ledger.capacity.active,'new_requests':self.ledger.new_requests,'confirmed_cost_usd':self.ledger.spent,'uncertain_reserve_usd':self.ledger.uncertain,'inflight_reservations_usd':self.ledger.inflight,'budget_usd':self.ledger.budget,'errors_count':len(self.errors),'recent_errors':self.errors[-8:],'source_counts':dict(self.source_counts),'stage_timings':timings,'filtering_started':False,'started_at':self.started,'updated_at':stamp(),'elapsed_seconds':time.monotonic()-self.started_mono,'startup_seconds':self.first_new_save_elapsed,'through_last_new_save_seconds':self.last_new_save_elapsed,'including_startup_documents_per_minute':(len(self.saved)-self.baseline)*60/self.last_new_save_elapsed if self.last_new_save_elapsed else None,'excluding_startup_documents_per_minute':(len(self.saved)-self.baseline-1)*60/(self.last_new_save_elapsed-self.first_new_save_elapsed) if self.first_new_save_elapsed is not None and self.last_new_save_elapsed>self.first_new_save_elapsed else None}
 async def monitor(self):
  last_success=0;last_ramp=time.monotonic()
  while not self.finished:
   await asyncio.to_thread(save,self.root/'raw-production-status.json',self.status('running'))
   cap=self.ledger.capacity
   if time.monotonic()-last_ramp>=30 and cap.successes-last_success>=16 and time.monotonic()-cap.last_backoff>=60:
    await cap.set_limit(cap.limit*2);last_success=cap.successes;last_ramp=time.monotonic()
   if (self.root/'stop-requested.json').exists():self.stop=True
   await asyncio.sleep(2)
 async def run(self):
  await self.initialize();monitor=asyncio.create_task(self.monitor());extractors=[asyncio.create_task(self.extractor()) for _ in range(self.topic_workers)];writers=[asyncio.create_task(self.writer()) for _ in range(self.writer_workers)]
  try:
   await asyncio.gather(*writers)
   self.stop=True
   # Let already dispatched topic requests finish and retain their responses; unblock producers waiting on a full queue.
   while any(not t.done() for t in extractors):
    while not self.ready.empty():self.ready.get_nowait()
    await asyncio.sleep(.1)
   await asyncio.gather(*extractors)
  finally:
   self.finished=True;await monitor
  state='target_reached' if len(self.saved)==self.target else 'stopped_requires_attention' if self.stop and (self.errors or self.ledger.halted) else 'seed_queue_exhausted'
  await asyncio.to_thread(save,self.root/'raw-production-status.json',self.status(state))
  # Export only; no text quality checks, rejection or filtering.
  def export():
   out=self.root/'raw-documents.jsonl';tmp=out.with_suffix('.tmp');n=0
   with tmp.open('w') as f:
    for p in sorted(self.out.glob('*.json')):f.write(json.dumps(json.loads(p.read_text()),ensure_ascii=False)+'\n');n+=1
    f.flush();os.fsync(f.fileno())
   assert n==len(self.saved);tmp.replace(out)
   save(self.root/'raw-export-receipt.json',{'documents':n,'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'filtering_started':False,'at':stamp()})
  await asyncio.to_thread(export);return self.status(state)

async def main(args):
 b=Path(args.base);r=b/'run';manifest=json.loads((r/'manifest.json').read_text());expected=json.loads((r/'raw-production-manifest.json').read_text())
 assert sha(Path(args.input).read_text())==expected['input_sha256']
 assert sha((Path(__file__).parent/'generator.json').read_text())==expected['generator_sha256']
 assert sha(Path(__file__).read_text())==expected['runner_sha256']
 rows=[json.loads(l) for l in Path(args.input).read_text().split('\n') if l.strip()];assert len(rows)==len({x['record_id'] for x in rows})
 limits=json.loads((r/'raw-production-limits.json').read_text());assert args.target==limits['target_raw_documents'] and args.budget==limits['budget_usd_cumulative']
 cap=Capacity(args.start_concurrency,args.max_concurrency)
 async with httpx.AsyncClient(headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY'],'Content-Type':'application/json'},timeout=httpx.Timeout(900,connect=20),limits=httpx.Limits(max_connections=128,max_keepalive_connections=128),follow_redirects=False) as client:
  endpoint=await client.get('https://openrouter.ai/api/v1/models/'+GENERATOR['model']+'/endpoints');endpoint.raise_for_status();flex=next(e for e in endpoint.json()['data']['endpoints'] if e['tag']=='openai/flex')
  for k,limit in [('prompt',.05),('completion',.25),('input_cache_write',.0625)]:assert float(flex['pricing'].get(k,0))*1e6<=limit+1e-10
  save(r/'raw-endpoint-verified.json',{'at':stamp(),'endpoint':flex})
  async def fetch(body):
   response=await client.post('https://openrouter.ai/api/v1/chat/completions',json=body)
   try:data=response.json()
   except ValueError:data={'error':{'type':'non_json_response'},'raw_response':response.text}
   return response.status_code,data,dict(response.headers)
  ledger=Ledger(r,args.budget,manifest['canonical_model'],cap,fetch);prod=Production(r,rows,ledger,args.target,args.topic_workers,args.max_concurrency)
  print(json.dumps(await prod.run()),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--input',required=True);p.add_argument('--target',type=int,default=10000);p.add_argument('--budget',type=float,default=20);p.add_argument('--start-concurrency',type=int,default=8);p.add_argument('--max-concurrency',type=int,default=64);p.add_argument('--topic-workers',type=int,default=8);a=p.parse_args()
 with (Path(a.base)/'run/worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(main(a))
