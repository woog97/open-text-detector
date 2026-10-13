"""Maximum-length, training-only safety verification of selected runtime settings."""
from profile_lora import *
from safetensors.torch import save_file
from safetensors import safe_open

def main():
 require_gpu();cfg=json.loads((ROOT/'configs/qwen35.json').read_text());info=json.loads((ROOT/'models.lock.json').read_text())['qwen35'];manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
 tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 records=json.loads((OUT/'verification-results.json').read_text()) if (OUT/'verification-results.json').exists() else []
 for stage,micro in [(1,4),(2,2)]:
  rows=read_data(f'stage{stage}-epoch0',manifest)
  encoded=[encode_example(r,tok,cfg['kind'],stage) for r in rows]
  selected=sorted(encoded,key=lambda e:len(e['ids']),reverse=True)[:32]
  for size,baseline in [(1,True)]+[(m,False) for m in ([4,2,1] if stage==1 else [2,1])]:
   existing=next((r for r in records if r['stage']==stage and r['baseline']==baseline and r['micro_batch']==size),None)
   if existing:
    if not baseline and existing.get('passed',False):break
    continue
   write('verification-status.json',{'stage':stage,'baseline':baseline,'state':'running','time':time.time()})
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   model=Detector.load_base(info,cfg['kind']);attach_lora(model,cfg);model.cuda().train()
   if baseline:model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
   else:model.backbone.gradient_checkpointing_disable()
   optimizer=bnb.optim.AdamW8bit(parameter_groups(model,cfg),lr=cfg['learning_rate'],weight_decay=.01)
   frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad};timings=[];bf16=True;finite=True
   torch.cuda.reset_peak_memory_stats();comparison=None
   for iteration in range(3):
    optimizer.zero_grad(set_to_none=True);start=sync();losses=[]
    for offset in range(0,32,size):
     batch=collate(selected[offset:offset+size],tok.pad_token_id)
     with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(batch);value,_=loss(outputs,batch,stage)
     bf16 &= all(v.dtype==torch.bfloat16 for v in outputs.values());losses.append(float(value.detach()));(value/(32//size)).backward()
    elapsed=sync()-start
    finite &= all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None and p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
    if iteration==0:
     path=OUT/f'stage{stage}-reference-gradients.safetensors'
     if baseline:save_file({n:p.grad.detach().cpu().contiguous() for n,p in model.named_parameters() if p.grad is not None},str(path))
     else:
      squared_error=reference_norm=candidate_norm=dot=0.
      with safe_open(path,framework='pt',device='cpu') as f:
       for n,p in model.named_parameters():
        if p.grad is None:continue
        a=p.grad.detach().float().cpu();b=f.get_tensor(n).float();squared_error+=float((a-b).square().sum());reference_norm+=float(b.square().sum());candidate_norm+=float(a.square().sum());dot+=float((a*b).sum());del a,b
      comparison={'relative_l2':(squared_error/reference_norm)**.5,'cosine':dot/(reference_norm*candidate_norm)**.5}
      print(json.dumps({'stage':stage,'micro_batch':size,'gradient_comparison':comparison}),flush=True)
    start=sync();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();elapsed+=sync()-start;timings.append(elapsed)
   frozen_ok=all(p._version==frozen[n] for n,p in model.named_parameters() if n in frozen)
   result={'stage':stage,'baseline':baseline,'micro_batch':size,'checkpointing':baseline,'sequence_lengths':[len(e['ids']) for e in selected],'step_seconds':timings,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'bf16_verified':bf16,'finite_gradients':finite,'frozen_verified':frozen_ok,'full_gradient_comparison':comparison}
   assert bf16 and finite and frozen_ok
   result['passed']=baseline or (result['peak_allocated_bytes']<.85*result['gpu_total_bytes'] and comparison['relative_l2']<.03 and comparison['cosine']>.999)
   records.append(result);write('verification-results.json',records);print(json.dumps(result),flush=True)
   if not baseline:del p
   del optimizer,model,outputs,value,batch;gc.collect();torch.cuda.empty_cache()
   if not baseline and result['passed']:break
 assert all(any(r['stage']==stage and not r['baseline'] and r['passed'] for r in records) for stage in [1,2])
 write('verification-status.json',{'state':'verified','time':time.time()})
if __name__=='__main__':main()
