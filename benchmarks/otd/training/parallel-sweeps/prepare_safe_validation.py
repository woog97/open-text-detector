from pathlib import Path
import json,sys,time,py_compile
r=Path('/data/workspace/paper-diversity-v1'); d=r/'auto-dispatch'; out=Path('/data/workspace/paper-batch-sweep-v1/throughput-profile/safe-validation-v1')
assert not out.exists(), 'Existing validation must be inspected, not overwritten'
out.mkdir()
source=r'''
import os
os.environ['HF_HUB_CACHE']='/data/workspace/model-cache'
os.environ['HF_HOME']='/data/workspace/hf-home'
import sys,json,time,random,heapq,gc,traceback,hashlib
from pathlib import Path
import torch,numpy as np
ROOT=Path('/data/workspace/paper-batch-sweep-v1/batch32-lr2e4');sys.path.insert(0,str(ROOT))
from train import require_gpu,read_data
from modeling import Detector,collate,loss
from adapters import attach_lora,parameter_groups
from data import encode_example
from transformers import AutoTokenizer
import bitsandbytes as bnb
OUT=Path(__file__).resolve().parent
def write(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False));t.replace(p)
def main():
 require_gpu();cfg=json.loads((ROOT/'configs/causal.json').read_text());info=json.loads((ROOT/'models.lock.json').read_text())['causal'];manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
 tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True);tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
 report={'purpose':'Longest actual epoch-zero training examples; BF16 gradient comparison and optimizer memory stress. Diagnostic only; no production config changes.','manifest_sha256':hashlib.sha256((ROOT/'prepared/manifest.json').read_bytes()).hexdigest(),'cases':[]}
 for stage,best in [(1,16),(2,8)]:
  write('status.json',{'state':'encoding_longest','stage':stage,'time':time.time()});heap=[];rows=read_data(f'stage{stage}-epoch0',manifest)
  for i,row in enumerate(rows):
   e=encode_example(row,tok,cfg['kind'],stage);item=(len(e['ids']),i,e)
   if len(heap)<32:heapq.heappush(heap,item)
   elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
  subset=[x[2] for x in sorted(heap)];assert len(subset)==32
  write(f'stage{stage}-data.json',{'scanned':len(rows),'indices':[x[1] for x in sorted(heap)],'lengths':[len(e['ids']) for e in subset]})
  reference=None
  for micro,checkpoint in [(4,True),(best,False)]:
   write('status.json',{'state':'validating','stage':stage,'micro':micro,'checkpointing':checkpoint,'time':time.time()})
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   model=Detector.load_base(info,cfg['kind']);attach_lora(model,cfg);model=model.cuda().train()
   if checkpoint:model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
   else:model.backbone.gradient_checkpointing_disable()
   optimizer=bnb.optim.AdamW8bit(parameter_groups(model,cfg),lr=cfg['learning_rate'],weight_decay=.01)
   frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad};values=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   for offset in range(0,32,micro):
    batch=collate(subset[offset:offset+micro],tok.pad_token_id)
    with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(batch);value,_=loss(outputs,batch,stage)
    assert all(v.dtype==torch.bfloat16 for v in outputs.values());assert torch.isfinite(value)
    values.append(float(value.detach()));(value/(32//micro)).backward()
   grads={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None};assert all(torch.isfinite(g).all() for g in grads.values())
   result={'stage':stage,'micro':micro,'checkpointing':checkpoint,'loss':sum(values)/len(values),'bf16_verified':True,'finite_gradients':True}
   if reference is None:reference=grads
   else:
    assert set(reference)==set(grads)
    dot=sum(float((g.double()*reference[n].double()).sum()) for n,g in grads.items());a=sum(float(g.double().square().sum()) for g in reference.values());b=sum(float(g.double().square().sum()) for g in grads.values());diff=sum(float((g.double()-reference[n].double()).square().sum()) for n,g in grads.items())
    result.update(gradient_cosine=dot/(a*b)**.5,gradient_relative_l2=(diff/a)**.5)
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
   assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
   result.update(frozen_verified=True,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
   report['cases'].append(result);write('results.json',report)
   del model,optimizer,outputs,value,batch,grads;gc.collect();torch.cuda.empty_cache()
 write('status.json',{'state':'complete','time':time.time(),'production_changes':False})
if __name__=='__main__':
 try:main()
 except Exception:
  write('status.json',{'state':'failed','time':time.time(),'traceback':traceback.format_exc()});raise
'''
p=out/'validate.py';p.write_text(source);py_compile.compile(str(p),doraise=True)
(out/'PROTOCOL.json').write_text(json.dumps({'scope':'Disposable Qwen3-0.6B models; compare micro4 checkpoint on with stage1 micro16/stage2 micro8 off on longest32 actual epoch0 examples; BF16 only; no production changes','cases':4,'data':'existing training only','new_downloads':False},indent=2))
(out/'READY').write_text('Syntax and protocol validated; no new dataset or production model changes.\n')
job={'id':'throughput-safe-validation-v1','rank':3,'enabled':True,'ready_file':str(out/'READY'),'completion_file':str(out/'status.json'),'dependencies':[],'command':[sys.executable,'-u',str(p)],'cwd':str(out)}
p=d/'jobs'/f"{job['id']}.json";assert not p.exists();t=p.with_suffix('.tmp');t.write_text(json.dumps(job,indent=2));t.replace(p)
print(json.dumps({'registered':job['id'],'source':str(out),'python':sys.executable}))
