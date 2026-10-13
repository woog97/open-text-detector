"""Allocate real AdamW8bit states and backpropagate without updating model weights."""
from runtime import require_space
import gc,json,time
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer
from train import ROOT,require_gpu,read_data,save
from modeling import Detector,collate,loss
from data import encode_example

def main():
 require_gpu();manifest=json.loads((ROOT/'prepared/manifest.json').read_text());lock=json.loads((ROOT/'models.lock.json').read_text());rows=read_data('stage2-epoch0',manifest);results={}
 # Exercise the installed optimizer kernel on a disposable standalone tensor, not a model.
 dummy=torch.nn.Parameter(torch.ones(8192,device='cuda'));dummy.grad=torch.ones_like(dummy);probe=bnb.optim.AdamW8bit([dummy],lr=1e-3);probe.step();assert torch.isfinite(dummy).all() and (dummy<1).all();del dummy,probe;torch.cuda.empty_cache()
 for key in ['encoder','causal','qwen35']:
  started=time.time();kind='encoder' if key=='encoder' else 'causal';info=lock[key];tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
  if tok.pad_token_id is None:tok.pad_token=tok.eos_token
  candidates=sorted(rows,key=lambda r:len(r['text']),reverse=True)[:1000];row=max(candidates,key=lambda r:len(tok(r['text'],add_special_tokens=False)['input_ids']));micro={'encoder':8,'causal':4,'qwen35':1}[key]
  model=Detector.load_base(info,kind).cuda();model.backbone.gradient_checkpointing_enable();model.train();torch.cuda.reset_peak_memory_stats()
  opt=bnb.optim.AdamW8bit(model.parameters(),lr=2e-5,weight_decay=.01);opt.check_overrides()
  for gi,group in enumerate(opt.param_groups):
   for pi,param in enumerate(group['params']):opt.init_state(group,param,gi,pi)
  versions={name:p._version for name,p in model.named_parameters()}
  batch=collate([encode_example(row,tok,kind,2)]*micro,tok.pad_token_id)
  with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(batch);value,parts=loss(outputs,batch,2)
  value.backward();assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert all(p._version==versions[name] for name,p in model.named_parameters());assert all(state['step']==0 for state in opt.state.values())
  peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
  assert peak<.9*total,'Insufficient peak memory headroom'
  results[key]={'optimizer':'bitsandbytes.AdamW8bit','bitsandbytes':bnb.__version__,'forward_dtype':str(outputs['tokens'].dtype),'actual_peak_with_allocated_optimizer_states_bytes':peak,'gpu_total_bytes':total,'headroom_bytes':total-peak,'finite_gradients':True,'model_optimizer_steps':0,'model_parameter_versions_unchanged':True,'toy_optimizer_kernel_check_passed':True,'elapsed_seconds':time.time()-started}
  save(ROOT/'optimizer-preflight-results.json',results);print(json.dumps({key:results[key]}),flush=True)
  del model,opt,batch,outputs,value,param,group;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
