"""Offline stage-2 checks. Never generates, retries, or deletes source data."""
import collections, concurrent.futures, datetime, hashlib, json, multiprocessing, os, sqlite3, sys, tarfile
from pathlib import Path
from mirror_core import assess, parse_topic, sha, words, writer_messages, PROTOCOL
BASE=Path('/tmp/pangram-luna-active-20261002')
OUT=BASE/'stage2-checked-30000-20261002'
ROWS={}; FINAL={}
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,value):
 p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(path):
 p=Path(path); r=json.loads(p.read_text()); flags=[];qc={};sid=r.get('source_record_id');parent=ROWS.get(sid)
 if not parent:flags.append('missing_parent')
 else:
  if sha(parent['text'])!=parent.get('passage_sha256') or parent.get('passage_sha256')!=r.get('source_passage_sha256') or FINAL.get(sid)!=r.get('source_passage_sha256'):flags.append('parent_hash_mismatch')
  for key in ['source_id','category','parent_document_id']:
   if r.get(key)!=parent.get(key):flags.append('lineage_'+key+'_mismatch')
  try:parse_topic(json.dumps({'topic':r.get('topic')}),parent['text'])
  except (ValueError,TypeError):flags.append('invalid_or_copied_topic')
  usage=r.get('usage') or {};reasoning=(usage.get('completion_tokens_details')or{}).get('reasoning_tokens');prompt=usage.get('prompt_tokens');completion=usage.get('completion_tokens')
  valid=all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in [prompt,completion,reasoning]) and completion>=reasoning
  if not valid:flags.append('visible_token_count_unverified')
  qc=assess(parent,r.get('text')or'',{'input_tokens':prompt if valid else 0,'output_tokens':completion-reasoning if valid else 0},r.get('finish_reason'));flags+=qc['flags']
  callpath=BASE/'run/calls'/(str(r.get('writer_call_id'))+'.json')
  if not callpath.exists():flags.append('writer_call_missing')
  else:
   call=json.loads(callpath.read_text());response=call.get('response')or{};choices=response.get('choices')or[]
   if call.get('source_record_id')!=sid or call.get('stage')!='writer':flags.append('writer_call_identity_mismatch')
   if call.get('http_status')!=200 or not choices:flags.append('writer_response_invalid')
   else:
    if (choices[0].get('message',{}).get('content')or'').strip()!=r.get('text'):flags.append('saved_response_text_mismatch')
    if choices[0].get('finish_reason')!=r.get('finish_reason') or response.get('usage')!=usage:flags.append('saved_response_metadata_mismatch')
   if call.get('request',{}).get('messages')!=writer_messages(parent,r.get('topic')):flags.append('writer_prompt_mismatch')
 if not r.get('text','').strip():flags.append('empty_output')
 return {'source_record_id':sid,'raw_file':p.name,'raw_sha256':digest(p),'normalized_output_sha256':sha(' '.join(words(r.get('text')or''))),'flags':sorted(set(flags)),'metrics':qc}
def main():
 global ROWS,FINAL
 OUT.mkdir(exist_ok=False)
 save('status.json',{'state':'checking','checked':0,'at':stamp()})
 status=json.loads((BASE/'run/raw-production-status.json').read_text())
 assert status['state']=='target_reached' and status['active_api_requests']==0
 paths=sorted((BASE/'run/raw-documents').glob('*.json'));assert len(paths)==30000
 with (BASE/'raw-production-30000-input.jsonl').open() as f:ROWS={r['record_id']:r for r in map(json.loads,f)}
 db=sqlite3.connect('file:/tmp/pangram-human-active-20261002/checkpoints/finish100k-active-20261002T193833Z.sqlite3?mode=ro',uri=True)
 for (raw,) in db.execute('select row from passages'):
  r=json.loads(raw);FINAL[r['record_id']]=r['passage_sha256']
 db.close()
 save('protocol.json',PROTOCOL)
 results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('fork')) as pool:
  for i,result in enumerate(pool.map(check,map(str,paths),chunksize=32),1):
   results.append(result)
   if i%1000==0:save('status.json',{'state':'checking','checked':i,'total':30000,'at':stamp()})
 seen=set();parents=set();accepted=[];counts=collections.Counter();cats=collections.Counter()
 with (OUT/'decisions.jsonl').open('w') as ledger,(OUT/'accepted.jsonl').open('w') as good,(OUT/'rejected.jsonl').open('w') as bad:
  for result,p in zip(results,paths):
   r=json.loads(p.read_text());assert digest(p)==result['raw_sha256']
   if result['source_record_id'] in parents:result['flags'].append('duplicate_parent')
   parents.add(result['source_record_id'])
   if not result['flags']:
    if result['normalized_output_sha256'] in seen:result['flags'].append('duplicate_output')
    else:seen.add(result['normalized_output_sha256'])
   result['passed']=not result['flags'];ledger.write(json.dumps(result)+'\n')
   r['stage2_automated_qc']=result;r['filtering_status']='passed' if result['passed'] else 'rejected';r['training_eligible']=False;r['admission_status']='stage2_checks_passed_parent_admission_pending' if result['passed'] else 'stage2_rejected'
   (good if result['passed'] else bad).write(json.dumps(r,ensure_ascii=False)+'\n')
   if result['passed']:accepted.append(str(p));cats[r['category']]+=1
   else:counts.update(result['flags'])
 save('status.json',{'state':'verifying_accepted','checked':30000,'accepted':len(accepted),'at':stamp()})
 # Repeat every per-output check on the accepted subset, then check export coverage and uniqueness.
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('fork')) as pool:
  for r in pool.map(check,accepted,chunksize=32):assert not r['flags'],r['source_record_id']
 accepted_ids=set();norms=set()
 with (OUT/'accepted.jsonl').open() as f:
  for r in map(json.loads,f):
   assert r['stage2_automated_qc']['passed'] and not r['stage2_automated_qc']['flags']
   assert r['source_record_id'] not in accepted_ids;accepted_ids.add(r['source_record_id'])
   norm=sha(' '.join(words(r['text'])));assert norm not in norms;norms.add(norm)
 assert len(accepted_ids)==len(accepted)
 rejected_count=sum(1 for _ in (OUT/'rejected.jsonl').open());assert rejected_count+len(accepted)==30000
 assert sorted(p.name for p in (BASE/'run/raw-documents').glob('*.json'))==[p.name for p in paths]
 assert all(digest(p)==r['raw_sha256'] for p,r in zip(paths,results))
 report={'at':stamp(),'state':'all_retained_pass_automated_stage2_checks','raw_count':30000,'accepted':len(accepted),'rejected':rejected_count,'rejection_flags_overlapping':dict(counts),'accepted_categories':dict(cats),'accepted_rechecked':len(accepted),'raw_files_unchanged':True,'new_model_calls':0,'replacement_generations':0,'model_mixture':'skipped_by_user','scope':'Existing stage-2 automated rules plus full parent/prompt/response linkage and normalized exact output deduplication','not_claimed':['semantic topic/genre review','stage-1 parent training admission','stage-5 final family/split readiness'],'artifacts':{name:{'sha256':digest(OUT/name),'bytes':(OUT/name).stat().st_size} for name in ['accepted.jsonl','rejected.jsonl','decisions.jsonl','protocol.json']}}
 save('report.json',report)
 receipt={'state':'complete','at':stamp(),'accepted':len(accepted),'rejected':rejected_count,'all_retained_checks_passed':True,'output_path':str(OUT),'backup':'not_uploaded; automatic approval review rejected proposed upload; original raw backup preserved'}
 save('receipt.json',receipt);save('status.json',receipt)
 print(json.dumps({'report':report,'receipt':receipt}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save('error.json',{'type':type(e).__name__,'at':stamp()})
  raise
