"""Matched training-only learning checks, with isolated fast kernels optional."""
import os,sys,json,time,random,gc,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
mode=int(sys.argv[1]);fast=bool(mode);stage=int(sys.argv[2]);micro=int(sys.argv[3])
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRITON_CACHE_DIR']=str(HERE/'triton-cache')
if fast:sys.path.insert(0,str(HERE/'kernel-vendor'))
sys.path.insert(0,str(ROOT))
from train import require_gpu,read_data
from modeling import Detector,collate,loss
from adapters import attach_lora,parameter_groups
from data import encode_example
from transformers import AutoTokenizer
import torch,numpy as np,bitsandbytes as bnb
from safetensors.torch import save_file
from safetensors import safe_open
OUT=HERE/'learning-check-v2';OUT.mkdir(exist_ok=True)
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def sync():torch.cuda.synchronize();return time.perf_counter()
@torch.inference_mode()
def assess(model,examples,pad):
 model.eval();values=[];probabilities=[]
 for e in examples:
  batch=collate([e],pad)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,_=loss(out,batch,stage)
  assert all(v.dtype==torch.bfloat16 for v in out.values());values.append(float(value));probabilities.extend(out['tokens'].float().softmax(-1)[0,:,1].cpu().tolist())
 return sum(values)/len(values),probabilities
require_gpu();cfg=json.loads((ROOT/'configs/qwen35.json').read_text());info=json.loads((ROOT/'models.lock.json').read_text())['qwen35'];manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
rows=read_data(f'stage{stage}-epoch0',manifest);encoded=[encode_example(r,tok,cfg['kind'],stage) for r in rows[:288]];training=encoded[:256];heldout=encoded[256:]
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
model=Detector.load_base(info,cfg['kind']);coverage=attach_lora(model,cfg);model.cuda();model.backbone.gradient_checkpointing_disable()
if mode==2:
 for module in model.backbone.modules():
  if isinstance(module,torch.nn.Linear) and not module.weight.requires_grad:
   module.weight.data=module.weight.data.to(torch.bfloat16)
   if module.bias is not None and not module.bias.requires_grad:module.bias.data=module.bias.data.to(torch.bfloat16)
optimizer=bnb.optim.AdamW8bit(parameter_groups(model,cfg),lr=cfg['learning_rate'],weight_decay=.01)
frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
initial,initial_probs=assess(model,heldout,tok.pad_token_id);timings=[];losses=[];finite=True;bf16=True;grad_comparison=None
torch.cuda.reset_peak_memory_stats()
for step in range(8):
 model.train();optimizer.zero_grad(set_to_none=True);start=sync();values=[]
 for offset in range(0,32,micro):
  batch=collate(training[step*32+offset:step*32+offset+micro],tok.pad_token_id)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,_=loss(out,batch,stage)
  bf16 &= all(v.dtype==torch.bfloat16 for v in out.values());finite &= bool(torch.isfinite(value));values.append(float(value.detach()));(value/(32//micro)).backward()
 forward_backward=sync()-start
 finite &= all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
 assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
 if step==0:
  reference=OUT/f'stage{stage}-baseline-gradients.safetensors'
  if not fast:save_file({n:p.grad.detach().cpu().contiguous() for n,p in model.named_parameters() if p.grad is not None},str(reference))
  else:
   error=bnorm=anorm=dot=0.
   with safe_open(reference,framework='pt',device='cpu') as f:
    for n,p in model.named_parameters():
     if p.grad is None:continue
     a=p.grad.detach().float().cpu();b=f.get_tensor(n).float();error+=float((a-b).square().sum());anorm+=float(a.square().sum());bnorm+=float(b.square().sum());dot+=float((a*b).sum());del a,b
   grad_comparison={'relative_l2':(error/bnorm)**.5,'cosine':dot/(anorm*bnorm)**.5}
 t=sync();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();timings.append(forward_backward+sync()-t);losses.append(sum(values)/len(values));print(json.dumps({'fast':fast,'stage':stage,'micro':micro,'step':step,'loss':losses[-1],'seconds':timings[-1]}),flush=True)
final,final_probs=assess(model,heldout,tok.pad_token_id)
result={'fast':fast,'mode':mode,'stage':stage,'micro_batch':micro,'effective_batch':32,'steps':8,'train_examples':256,'training_only_holdout_examples':32,'initial_loss':initial,'final_loss':final,'initial_probabilities':initial_probs,'final_probabilities':final_probs,'train_losses':losses,'step_seconds':timings,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'finite_gradients':finite,'bf16_verified':bf16,'frozen_verified':all(p._version==frozen[n] for n,p in model.named_parameters() if n in frozen),'gradient_comparison':grad_comparison,'data_manifest_sha256':hashlib.sha256((ROOT/'prepared/manifest.json').read_bytes()).hexdigest()}
assert finite and bf16 and result['frozen_verified']
save(f'stage{stage}-'+('fast-bf16' if mode==2 else 'fast' if fast else 'baseline')+f'-micro{micro}.json',result)
