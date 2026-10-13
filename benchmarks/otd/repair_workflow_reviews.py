"""Bounded mechanical evidence-quote repair; never regenerates or selects writing."""
import asyncio,fcntl,json
import paper_workflow_eval as w

async def main():
 lock=(w.OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((w.OUT/'protocol.json').read_text())
 assert protocol['judge_prompt']==w.JUDGE
 client=await w.initialize();jobs={j['id']:j for p in w.rows('passages.jsonl') for j in w.build_jobs(p)}
 failed=[]
 for r in w.rows('failures.jsonl'):
  if 'Mechanical retry budget exhausted: ' in r['error']:
   rid=r['error'].split('Mechanical retry budget exhausted: ',1)[1]
   if rid.endswith('/judge') and rid not in client.done and rid not in failed:failed.append(rid)
 for rid in failed:
  j=jobs[rid.removesuffix('/judge')];candidate=client.done[j['id']+'/writer']['output']['paragraph']
  attempts=[a for a in w.rows('attempts.jsonl') if a['request_id']==rid]
  last=attempts[-1]['response'];obj=json.loads(last['choices'][0]['message']['content']);invalid=[]
  for i,d in enumerate(obj.get('differences',[])):
   for field,text in [('original_quote',j['original']),('candidate_quote',candidate)]:
    if isinstance(d.get(field),str) and d[field] not in text:invalid.append({'field':f'differences[{i}].{field}','invalid_quote':d[field]})
  assert invalid,'This recovery path is only for literal evidence-quote errors'
  repair_id=rid+'/evidence-format-repair'
  prompt=w.JUDGE+'\nMechanical evidence validation failed previously for these fields: '+json.dumps(invalid,ensure_ascii=False)+'. Copy SHORT literal substrings including their exact punctuation and spacing. No paraphrased evidence or inserted ellipses. Evaluate the same original and candidate; do not revise the candidate.'
  payload={'abstract':j['abstract'],'text_before':j['prefix'],'original_target':j['original'],'candidate_target':candidate,'text_after':j['suffix']}
  result=await client.call(repair_id,'judge',prompt,payload,lambda text,j=j,c=candidate:w.parse_judge(text,j,c))
  actual=client.done[repair_id]
  row={'request_id':rid,'stage':'judge','output':result,'generation_id':actual['generation_id'],'mechanical_repair_request_id':repair_id}
  w.append('responses.jsonl',row);client.done[rid]=row
  w.append('review-repairs.jsonl',{'at_utc':w.b.now(),'request_id':rid,'repair_request_id':repair_id,'original_candidate_sha256':w.b.sha(candidate),'original_attempts':len(attempts),'extra_attempt_cap':3,'reason':'Literal evidence-quote validation only; writing and evaluation rubric unchanged'})
  print('Repaired literal evidence validation: '+rid,flush=True)
 w.save('costs.json',w.accounting())
if __name__=='__main__':asyncio.run(main())
