"""Maximum-length BF16 optimizer-step memory validation, on disposable models."""
import sys,os,json,time,gc,random
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRITON_CACHE_DIR']=str(HERE/'triton-cache')
sys.path.insert(0,str(HERE/'kernel-vendor'));sys.path.insert(0,str(ROOT))
from train import require_gpu,read_data
from modeling import Detector,collate,loss
from adapters import attach_lora,parameter_groups
from data import encode_example
from transformers import AutoTokenizer
import torch,numpy as np,bitsandbytes as bnb
frozen_storage=bool(int(sys.argv[1])) if len(sys.argv)>1 else True
OUT=HERE/('stress-v2' if frozen_storage else 'stress-v2-fp32');OUT.mkdir(exist_ok=True)
def sync():torch.cuda.synchronize();return time.perf_counter()
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
require_gpu();cfg=json.loads((ROOT/'configs/qwen35.json').read_text());info=json.loads((ROOT/'models.lock.json').read_text())['qwen35'];manifest=json.loads((ROOT/'prepared/manifest.json').read_text());selected=json.loads((HERE/'learning-check-v2/selected.json').read_text())
tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
for stage in [1,2]:
 micro=selected[str(stage)];save('status.json',{'state':'running','stage':stage,'micro':micro,'time':time.time()})
 rows=read_data(f'stage{stage}-epoch0',manifest);examples=sorted([encode_example(r,tok,cfg['kind'],stage) for r in rows],key=lambda e:len(e['ids']),reverse=True)[:32]
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 model=Detector.load_base(info,cfg['kind']);attach_lora(model,cfg);model.cuda().train();model.backbone.gradient_checkpointing_disable()
 for module in model.backbone.modules():
  if frozen_storage and isinstance(module,torch.nn.Linear) and not module.weight.requires_grad:
   module.weight.data=module.weight.data.to(torch.bfloat16)
   if module.bias is not None and not module.bias.requires_grad:module.bias.data=module.bias.data.to(torch.bfloat16)
 optimizer=bnb.optim.AdamW8bit(parameter_groups(model,cfg),lr=cfg['learning_rate'],weight_decay=.01);frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
 torch.cuda.reset_peak_memory_stats();timings=[];bf16=True;finite=True
 for step in range(5):
  optimizer.zero_grad(set_to_none=True);start=sync()
  for i in range(0,32,micro):
   batch=collate(examples[i:i+micro],tok.pad_token_id)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,_=loss(out,batch,stage)
   bf16 &= all(v.dtype==torch.bfloat16 for v in out.values());finite &= bool(torch.isfinite(value));(value/(32//micro)).backward()
  finite &= all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
  assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();timings.append(sync()-start)
 result={'stage':stage,'micro_batch':micro,'effective_batch':32,'sequence_lengths':[len(e['ids']) for e in examples],'step_seconds':timings,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'bf16_verified':bf16,'finite_gradients':finite,'frozen_verified':all(p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)}
 assert bf16 and finite and result['frozen_verified'] and result['peak_allocated_bytes']<.85*result['gpu_total_bytes']
 save(f'stage{stage}.json',result);print(json.dumps(result),flush=True);del model,optimizer,out,value,batch,module;gc.collect();torch.cuda.empty_cache()
save('status.json',{'state':'verified','time':time.time()})
