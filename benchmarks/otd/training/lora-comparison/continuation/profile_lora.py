"""Disposable BF16 Qwen35 throughput trials; never opens the production gate."""
import sys,json,time,hashlib,traceback,gc,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train import require_gpu,read_data
from modeling import Detector,collate,loss
from adapters import attach_lora,parameter_groups
from data import encode_example
from transformers import AutoTokenizer
import torch,numpy as np,bitsandbytes as bnb
OUT=ROOT/'continuation/profiling';OUT.mkdir(exist_ok=True)
def write(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sync():torch.cuda.synchronize();return time.perf_counter()
def main():
 require_gpu();cfg=json.loads((ROOT/'configs/qwen35.json').read_text());info=json.loads((ROOT/'models.lock.json').read_text())['qwen35'];manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
 tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision']);tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
 for stage in [1,2]:
  rows=read_data(f'stage{stage}-epoch0',manifest)[:512]
  start=time.perf_counter();encoded=[encode_example(r,tok,cfg['kind'],stage) for r in rows];encode_seconds=time.perf_counter()-start
  ordered=sorted(range(len(rows)),key=lambda i:len(encoded[i]['ids']))
  indices=[ordered[round(i*(len(ordered)-1)/63)] for i in range(64)]
  selected=[encoded[i] for i in indices];raw=[rows[i] for i in indices]
  write(f'stage{stage}-data.json',{'indices':indices,'lengths':[len(e['ids']) for e in selected],'encode_seconds_512':encode_seconds,'data_manifest_sha256':hashlib.sha256((ROOT/'prepared/manifest.json').read_bytes()).hexdigest()})
  for micro,checkpoint in [(1,True),(2,True),(4,True),(1,False),(2,False),(4,False)]:
   name=f'stage{stage}-micro{micro}-checkpoint{int(checkpoint)}';path=OUT/(name+'.json')
   if path.exists():continue
   write('status.json',{'state':'profiling','case':name,'time':time.time()})
   model=optimizer=None
   try:
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=Detector.load_base(info,cfg['kind']);coverage=attach_lora(model,cfg);model=model.cuda().train()
    if checkpoint:model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.backbone.gradient_checkpointing_disable()
    optimizer=bnb.optim.AdamW8bit(parameter_groups(model,cfg),lr=cfg['learning_rate'],weight_decay=.01)
    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    timings=[];signature=None;loss_signature=None;bf16=True;finite=True;frozen_ok=True
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(3):
     optimizer.zero_grad(set_to_none=True);timing={'prepare':0.,'forward':0.,'backward':0.,'optimizer':0.};values=[]
     subset=selected[:32] if iteration==0 else selected[32:]
     wall=sync()
     for offset in range(0,32,micro):
      t=sync();batch=collate(subset[offset:offset+micro],tok.pad_token_id);timing['prepare']+=sync()-t
      t=sync()
      with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(batch);value,_=loss(outputs,batch,stage)
      timing['forward']+=sync()-t;bf16 &= all(v.dtype==torch.bfloat16 for v in outputs.values());finite &= bool(torch.isfinite(value));values.append(float(value.detach()))
      t=sync();(value/(32//micro)).backward();timing['backward']+=sync()-t
     finite &= all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
     frozen_ok &= all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
     if iteration==0:
      signature={n:p.grad.detach().flatten()[::max(1,p.numel()//32)][:32].float().cpu().tolist() for n,p in model.named_parameters() if p.grad is not None}
      loss_signature=sum(values)/len(values)
      assert all(p.grad is not None for n,p in model.named_parameters() if '.lora_B.' in n)
      assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if '.lora_B.' in n and 'mlp' in n)
     t=sync();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();timing['optimizer']+=sync()-t;timing['wall']=sync()-wall;timings.append(timing)
     del outputs,value,batch
    frozen_ok &= all(p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
    # Measure exact equivalent CPU preprocessing with cached and uncached rows.
    t=time.perf_counter();again=[encode_example(r,tok,cfg['kind'],stage) for r in raw];uncached=time.perf_counter()-t
    assert again==selected
    measured=sum(x['wall'] for x in timings[1:])/2
    result={'case':name,'stage':stage,'micro_batch':micro,'checkpointing':checkpoint,'timings':timings,'warmup_steps':1,'measured_steps':2,'seconds_per_32_examples':measured,'examples_per_second':32/measured,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'bf16_verified':bf16,'finite_gradients':finite,'frozen_verified':frozen_ok,'initial_loss':loss_signature,'gradient_signature':signature,'uncached_encode_seconds_64':uncached,'cached_encoding_exact':True,'coverage':coverage}
    assert bf16 and finite and frozen_ok
    write(name+'.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['gradient_signature','coverage']}),flush=True)
   except torch.cuda.OutOfMemoryError:
    write(name+'.json',{'case':name,'status':'out_of_memory'});print(name+' OOM',flush=True)
   except Exception:
    write(name+'.json',{'case':name,'status':'error','traceback':traceback.format_exc()});raise
   finally:
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
 write('status.json',{'state':'measurements_complete','time':time.time()})
if __name__=='__main__':main()
