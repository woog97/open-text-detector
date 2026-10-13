"""Actual-backbone BF16 forward/backward validation; ZERO optimizer steps/checkpoints."""
from runtime import require_space, SPACE_CACHE
import argparse,json,time,gc
from pathlib import Path
import torch
from transformers import AutoTokenizer
from data import encode_example
from modeling import Detector,collate,loss
from train import ROOT,require_gpu,read_data,save

def main(a):
 require_gpu();manifest=json.loads((ROOT/'prepared/manifest.json').read_text());lock=json.loads((ROOT/'models.lock.json').read_text());rows=read_data('stage2-epoch0',manifest);results={}
 for key in a.models.split(','):
  kind='encoder' if key=='encoder' else 'causal'
  started=time.time();info=lock[key];tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'])
  if tok.pad_token_id is None:tok.pad_token=tok.eos_token
  candidates=sorted(rows,key=lambda r:len(r['text']),reverse=True)[:1000]
  longest=max(candidates,key=lambda r:len(tok(r['text'],add_special_tokens=False)['input_ids']))
  micro={'encoder':8,'causal':4,'qwen35':1}[key]
  probe_rows=[longest]*micro
  torch.manual_seed(42);torch.cuda.reset_peak_memory_stats();model=Detector.load_base(info,kind).cuda();model.backbone.gradient_checkpointing_enable();model.train();checks=[]
  for stage in [1,2]:
   examples=[encode_example(r,tok,kind,stage) for r in probe_rows];batch=collate(examples,tok.pad_token_id);model.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,parts=loss(out,batch,stage)
   assert torch.isfinite(value);assert out['tokens'].dtype==torch.bfloat16;value.backward()
   grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
   if stage==2:assert model.token_head.weight.grad.abs().sum()>0
   checks.append({'stage':stage,'loss':float(value.detach()),'forward_dtype':str(out['tokens'].dtype),'finite_gradients':True,'input_lengths':batch['attention_mask'].sum(1).tolist(),'source_lengths':[len(e['source_labels']) for e in examples]})
  model.zero_grad(set_to_none=True);model.eval();visibility={}
  if kind=='causal':
   ids=tok('The first claim concerns optimization. Later evidence changes the conclusion.',add_special_tokens=False)['input_ids'];altered=ids[:];altered[-2]=tok(' however',add_special_tokens=False)['input_ids'][0]
   tensor=torch.tensor([ids+ids,altered+altered],device='cuda');mask=torch.ones_like(tensor)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):h=model.backbone(input_ids=tensor,attention_mask=mask).last_hidden_state
   first=float((h[0,0].float()-h[1,0].float()).abs().max());second=float((h[0,len(ids)].float()-h[1,len(ids)].float()).abs().max());assert first<1e-5 and second>0
   visibility={'first_copy_prefix_delta':first,'second_copy_prefix_delta':second,'causal_mask_preserved':True,'second_copy_sees_later_source_content':True}
  results[key]={'model':info,'parameters':sum(p.numel() for p in model.parameters()),'checks':checks,'repeat2_visibility':visibility,'elapsed_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'master_parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),'estimated_peak_with_adam_bytes':torch.cuda.max_memory_allocated()+sum(p.numel()*8 for p in model.parameters()),'optimizer_steps':0,'trained_checkpoint_saved':False}
  save(ROOT/'preflight-results.json',results);print(json.dumps({key:results[key]}),flush=True);del model,out,batch,grads,value;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',default='encoder,causal,qwen35');main(p.parse_args())
