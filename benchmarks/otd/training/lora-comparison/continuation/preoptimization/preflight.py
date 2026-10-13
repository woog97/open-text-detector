"""Deferred until the baseline queue completes. BF16 real-backbone checks, no optimizer steps."""
from runtime import require_space
import json,gc,hashlib,argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file,load_file
from adapters import attach_lora,parameter_groups
from modeling import Detector,collate,loss
from data import encode_example
from train import ROOT,require_gpu,read_data,save

def main(models):
 require_gpu();manifest=json.loads((ROOT/'prepared/manifest.json').read_text());lock=json.loads((ROOT/'models.lock.json').read_text());rows=read_data('stage2-epoch0',manifest);results={}
 for key in models:
  cfg=json.loads((ROOT/'configs'/(key+'.json')).read_text());info=lock[key]
  tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
  if tok.pad_token_id is None:tok.pad_token=tok.eos_token
  candidates=sorted(rows,key=lambda r:len(r['text']),reverse=True)[:1000]
  longest=max(candidates,key=lambda r:len(tok(r['text'],add_special_tokens=False)['input_ids']))
  torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();model=Detector.load_base(info,cfg['kind']);coverage=attach_lora(model,cfg);parameter_groups(model,cfg);model.cuda();model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train();checks=[]
  for stage in [1,2]:
   batch=collate([encode_example(longest,tok,cfg['kind'],stage)]*cfg['micro_batch'],tok.pad_token_id);model.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,_=loss(out,batch,stage)
   assert out['tokens'].dtype==torch.bfloat16 and torch.isfinite(value);value.backward()
   assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
   bgrads={n:p.grad for n,p in model.backbone.named_parameters() if '.lora_B.' in n}
   assert bgrads and all(g is not None and torch.isfinite(g).all() for g in bgrads.values())
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   assert sum(float(g.abs().sum()) for n,g in bgrads.items() if '.mlp.' in n)>0
   checks.append({'stage':stage,'finite_adapter_gradients':True,'frozen_base_has_no_gradients':True,'mlp_gradients_nonzero':True,'forward_dtype':str(out['tokens'].dtype),'loss':float(value.detach())})
  # Round-trip trainable state after a deterministic nonzero adapter perturbation.
  model.zero_grad(set_to_none=True);model.eval()
  with torch.no_grad():
   for n,p in model.backbone.named_parameters():
    if '.lora_B.' in n:p.add_(0.0001)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):expected=model(batch)['tokens'].clone()
  state={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
  tmp=ROOT/'preflight-adapter.tmp.safetensors';save_file(state,str(tmp))
  with torch.no_grad():
   for p in model.parameters():
    if p.requires_grad:p.zero_()
  restored=load_file(str(tmp),device='cuda');missing,unexpected=model.load_state_dict(restored,strict=False)
  assert not unexpected and not any(n in restored for n in missing)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(batch)['tokens']
  assert torch.equal(expected,actual);tmp.unlink()
  results[key]={'model':info,'coverage':coverage,'checks':checks,'adapter_roundtrip_exact_bf16':True,'optimizer_steps':0,'peak_gpu_bytes':torch.cuda.max_memory_allocated()};save(ROOT/'preflight-results.json',results)
  print(json.dumps({'model':key,'passed':True,'target_count':coverage['target_count']}),flush=True)
  del model,out,value,batch,bgrads,expected,actual,state,restored;gc.collect();torch.cuda.empty_cache()
 # Freeze source, configs, and the SAME prepared manifest only once preflight passed.
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 save(ROOT/'READY.json',{'status':'prepared_not_trained','data_manifest_sha256':digest(ROOT/'prepared/manifest.json'),'source_code_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')},'configs_sha256':{p.name:digest(p) for p in (ROOT/'configs').glob('*.json')},'preflight_sha256':digest(ROOT/'preflight-results.json')})
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--models',default='encoder,causal,qwen35');main(parser.parse_args().models.split(','))
