"""BF16-only Space mirror pilot with durable per-call provenance and no API use."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl,json,os,random,time
from pathlib import Path
from mirror_core import PROTOCOL,validate_parent,topic_messages,parse_topic,writer_messages,assess,sha

def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def stamp():return datetime.now(timezone.utc).isoformat()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True)
 p.add_argument('--pilot',action='store_true');p.add_argument('--limit',type=int,default=12);p.add_argument('--seed',type=int,default=27183)
 args=p.parse_args()
 if not str(args.model_path.resolve()).startswith('/data/workspace/'):p.error('Model assets must be on the training Space')
 rows=[json.loads(x) for x in args.input.read_text().splitlines() if x.strip()][:args.limit]
 if not rows:raise ValueError('No source inputs')
 for row in rows:validate_parent(row,args.pilot)
 args.out.mkdir(parents=True,exist_ok=True);lock=(args.out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,Qwen3_5ForConditionalGeneration
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA is required')
 torch.set_num_threads(4)
 tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
 config=AutoConfig.from_pretrained(str(args.model_path),local_files_only=True)
 cls=Qwen3_5ForConditionalGeneration if config.model_type=='qwen3_5' else AutoModelForCausalLM
 model=cls.from_pretrained(str(args.model_path),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
 if any(p.is_floating_point() and p.dtype!=torch.bfloat16 for p in model.parameters()):raise RuntimeError('Model parameters are not all BF16')
 model.generation_config.do_sample=True
 protocol_hash=sha(json.dumps(PROTOCOL,sort_keys=True))
 identity={'protocol_sha256':protocol_hash,'input_sha256':sha(args.input.read_text()),'model_path':str(args.model_path),'model_revision':args.model_path.name,'pilot':args.pilot,'seed':args.seed,'limit':args.limit,'dtype':'bfloat16'}
 manifest=args.out/'manifest.json'
 if manifest.exists() and json.loads(manifest.read_text())!=identity:raise RuntimeError('Resume configuration mismatch')
 atomic(manifest,identity);atomic(args.out/'protocol.json',PROTOCOL)
 def call(messages,stage,row_id,max_tokens,temperature):
  call_id=sha(row_id+stage+json.dumps(messages,sort_keys=True)+json.dumps(identity,sort_keys=True))
  dest=args.out/'calls'/f'{call_id}.json';dest.parent.mkdir(exist_ok=True)
  if dest.exists():return json.loads(dest.read_text())
  seed=int(call_id[:8],16);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  batch=tokenizer(prompt,return_tensors='pt').to('cuda');n=batch['input_ids'].shape[-1]
  started=time.time()
  with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   output=model.generate(**batch,max_new_tokens=max_tokens,do_sample=temperature>0,temperature=temperature if temperature>0 else None,top_p=.95 if temperature>0 else None,pad_token_id=tokenizer.eos_token_id)
  ids=output[0,n:];text=tokenizer.decode(ids,skip_special_tokens=True).strip()
  eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
  finish='stop' if len(ids) and int(ids[-1]) in eos else 'length'
  answer={'call_id':call_id,'source_record_id':row_id,'stage':stage,'messages':messages,'response':text,'usage':{'input_tokens':n,'output_tokens':len(ids)},'finish_reason':finish,'seed':seed,'temperature':temperature,'max_new_tokens':max_tokens,'elapsed_seconds':time.time()-started,'inference_dtype':'bfloat16','created_at':stamp(),**identity}
  atomic(dest,answer);return answer
 records=[]
 for i,row in enumerate(rows):
  dest=args.out/'records'/f"{row['record_id']}.json";dest.parent.mkdir(exist_ok=True)
  if dest.exists():records.append(json.loads(dest.read_text()));continue
  common={'mirror_id':sha(row['record_id']+json.dumps(identity,sort_keys=True)),'source_record_id':row['record_id'],'source_passage_sha256':row['passage_sha256'],'source_id':row['source_id'],'category':row['category'],
   'parent_document_id':row.get('parent_document_id'),'document_family_id':row.get('document_family_id',row.get('provisional_family_id')),'split':row.get('split') if not args.pilot else 'development_pilot',
   'development_exposed':args.pilot,'parent_admission_status':row.get('admission_status'),'admission_status':'quarantined_synthetic_candidate','training_eligible':False,'known_process':'independent_ai_generation','review_status':'pending',**identity}
  topic_call=call(topic_messages(row),'topic',row['record_id'],180,0)
  try:
   if topic_call['finish_reason']!='stop':raise ValueError('Topic generation incomplete')
   topic=parse_topic(topic_call['response'],row['text'])
  except ValueError as exc:
   result={**common,'qc':{'passed':False,'flags':['invalid_topic'],'detail':str(exc)},'topic_call_id':topic_call['call_id']}
  else:
   writer_call=call(writer_messages(row,topic),'writer',row['record_id'],min(4000,max(320,int(len(row['text'].split())*2.4))),.8)
   qc=assess(row,writer_call['response'],writer_call['usage'],writer_call['finish_reason'])
   result={**common,'topic':topic,'text':writer_call['response'],'text_sha256':sha(writer_call['response']),'topic_call_id':topic_call['call_id'],'writer_call_id':writer_call['call_id'],'qc':qc}
  atomic(dest,result);records.append(result)
  summary={'state':'running','completed':len(records),'planned':len(rows),'mechanically_passed':sum(r['qc']['passed'] for r in records),'admitted':0,'updated_at':stamp()}
  atomic(args.out/'status.json',summary);print(json.dumps(summary),flush=True)
 (args.out/'mirrors.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 summary.update(state='complete',rejection_reasons=dict(Counter(flag for r in records for flag in r['qc']['flags'])),api_cost_usd=0,model=str(args.model_path),inference_dtype='bfloat16',updated_at=stamp())
 atomic(args.out/'status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
