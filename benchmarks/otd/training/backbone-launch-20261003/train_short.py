"""Current binary recipe across authorized backbones; metrics-only W&B; no final eval."""
from runtime import ROOT,require_space
import argparse,json,gzip,hashlib,random,time,math,os,traceback
from pathlib import Path
import numpy as np
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from safetensors.torch import save_file,load_file
from data import encode_example
from modeling import Detector,collate,loss,training_loss
from adapters_short import attach_lora,parameter_groups

def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def rows(root,key,manifest):
 b=gzip.decompress((root/'prepared-v2'/f'{key}.jsonl.gz').read_bytes());assert hashlib.sha256(b).hexdigest()==manifest['files'][key]['sha256'];rs=[json.loads(l) for l in b.splitlines()]
 if key.startswith('stage'):
  from collections import Counter
  target={k:v//10 for k,v in manifest['counts'][key].items()};seen=Counter();short=[]
  for row in rs:
   ds=row['dataset']
   if seen[ds]<target[ds]:short.append(row);seen[ds]+=1
  assert dict(seen)==target
  return short
 return rs
def batches(rs,tok,kind,stage,size):
 for i in range(0,len(rs),size):yield collate([encode_example(r,tok,kind,stage) for r in rs[i:i+size]],tok.pad_token_id)
def checkpoint(model,path):
 tensors={n:p.detach().to(device='cpu',dtype=torch.bfloat16).contiguous() for n,p in model.named_parameters() if p.requires_grad}
 assert all(t.dtype==torch.bfloat16 for t in tensors.values())
 tmp=path.with_suffix('.tmp');save_file(tensors,str(tmp));tmp.replace(path)
def restore(model,path):
 d=load_file(str(path));expected={n for n,p in model.named_parameters() if p.requires_grad};assert set(d)==expected
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in d:p.copy_(d[n])
@torch.inference_mode()
def validate(model,tok,rs,stage,size):
 model.eval();total=0;n=0
 for b in batches(rs,tok,model.kind,stage,size):
  with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b);v,_=loss(o,b,stage)
  assert torch.isfinite(v);total+=float(v)*b['logical_size'];n+=b['logical_size']
 return total/n

def main(name):
 require_space();assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
 spec=next(m for m in json.loads((ROOT/'models.json').read_text()) if m['name']==name)
 root=ROOT/'runs'/name;out=root/'run';out.mkdir(exist_ok=True)
 assert not (out/'run.json').exists(),'Existing training run: do not restart'
 cfg={'kind':spec['kind'],'training_fraction':0.1,'total_training_examples':8400,'seed':42,'learning_rate':2e-4,'head_learning_rate':2e-5,'micro_batch':4 if spec['kind']=='encoder' else 1,'effective_batch':32,'stages':{'1':{'epochs':1},'2':{'epochs':3}},'lora':{'rank':spec.get('lora_rank',128),'expert_rank':spec.get('expert_rank',128),'alpha':32},'source_tokens_max':510,'precision':'BF16','trainable_dtype':'bfloat16','checkpoint_dtype':'bfloat16','optimizer':'adamw8bit'}
 for seed in [cfg['seed']]:random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 asset=ROOT/'assets'/name;tok=AutoTokenizer.from_pretrained(asset,local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 manifest=json.loads((root/'prepared-v2/manifest.json').read_text());save(out/'status.json',{'state':'loading_model'})
 model=Detector.load_base({'repo':str(asset),'revision':None},cfg['kind']);coverage=attach_lora(model,cfg)
 assert all(p.dtype==torch.bfloat16 for p in model.parameters() if p.requires_grad)
 if len(spec.get('gpus',[]))>1:
  from accelerate import infer_auto_device_map,dispatch_model
  layer_types=list({m.__class__.__name__ for n,m in model.named_modules() if n.startswith('backbone.layers.') and n.count('.')==2})
  device_map=infer_auto_device_map(model,max_memory={0:'60GiB',1:'60GiB'},no_split_module_classes=layer_types)
  assert set(device_map.values())=={0,1},device_map
  model=dispatch_model(model,device_map=device_map)
  coverage['device_map']=device_map
 else:model=model.cuda()
 model.physical_microbatch=2 if cfg['kind']=='encoder' else 1
 model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 if hasattr(model.backbone,'enable_input_require_grads'):model.backbone.enable_input_require_grads()
 save(out/'run.json',{'config':cfg,'model':spec,'assets':str(asset),'precision':'BF16 forwards, frozen weights, trainable adapters, classifier heads and checkpoints; FP32 loss/metric calculations','adapter_coverage':coverage,'data_manifest_sha256':hashlib.sha256((root/'prepared-v2/manifest.json').read_bytes()).hexdigest(),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'checkpoint_format':'trainable adapters and classifier heads; pinned base stored separately'})
 # Real reduced-precision preflight: document labels never become token gold.
 rs=rows(root,'stage2-epoch0',manifest);paper=next(r for r in rs if r.get('supervision')!='document_only');doc=next(r for r in rs if r.get('supervision')=='document_only')
 for label,r in [('paper',paper),('document',doc)]:
  model.zero_grad(set_to_none=True);b=collate([encode_example(r,tok,cfg['kind'],2)],tok.pad_token_id);model.train()
  with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b);v,_=training_loss(o,b,2)
  assert o['tokens'].dtype==torch.bfloat16 and torch.isfinite(v);v.backward()
  assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.backbone.named_parameters() if 'lora_B' in n)
  if label=='document':assert all(p.grad is None or not p.grad.any() for h in [model.token_head,model.segment_head,model.mixed_head] for p in h.parameters())
  del b,o,v
 model.zero_grad(set_to_none=True);torch.cuda.empty_cache();save(root/'preflight.json',{'passed':True,'bf16':True,'frozen_weights_unchanged':True,'document_only_no_localization_gradients':True})
 # Same metrics-only payload as the approved tracker: no configuration or machine metadata.
 import wandb
 settings=wandb.Settings(console='off',disable_code=True,disable_git=True,save_code=False,disable_job_creation=True,x_disable_meta=True,x_disable_stats=True,x_disable_machine_info=True)
 wr=wandb.init(entity='rigg-alice0',project='open-text-detector-classifiers',group='text-classifiers',name=name+'-current-mix-10pct-20261003',config={},settings=settings,dir=str(root))
 save(root/'wandb-tracking.json',{'run_id':wr.id,'url':wr.url,'project':'open-text-detector-classifiers','group':'text-classifiers'})
 selection=rows(root,'selection-windows',manifest);history=[];started=time.time();global_step=0;processed=0;done_rows=0
 try:
  for stage in [1,2]:
   epochs=cfg['stages'][str(stage)]['epochs'];size=cfg['micro_batch'];accum=cfg['effective_batch']//size;best=float('inf')
   opt=bnb.optim.AdamW8bit(parameter_groups(model,cfg),weight_decay=.01)
   total_steps=sum(math.ceil(manifest['files'][f'stage{stage}-epoch{e}']['rows']/10/cfg['effective_batch']) for e in range(epochs));sched=get_cosine_schedule_with_warmup(opt,math.ceil(.06*total_steps),total_steps)
   for epoch in range(epochs):
    rs=rows(root,f'stage{stage}-epoch{epoch}',manifest);model.train();opt.zero_grad(set_to_none=True);n=math.ceil(len(rs)/size);running=0;epochstart=time.time()
    for i,b in enumerate(batches(rs,tok,cfg['kind'],stage,size)):
     count=min(accum,n-(i//accum)*accum)
     with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b);v,parts=training_loss(pred,b,stage)
     if not torch.isfinite(v):raise RuntimeError('Nonfinite training loss')
     (v/count).backward();running+=float(v.detach());processed+=int(b['attention_mask'].sum());done_rows+=b['logical_size']
     if (i+1)%accum==0 or i+1==n:
      torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();opt.zero_grad(set_to_none=True);global_step+=1
     if i%max(1,32//size)==0:
      status={'state':'training','stage':stage,'epoch':epoch,'batch':i,'batches':n,'loss':float(v.detach()),'elapsed_seconds':time.time()-started,'optimizer_steps':global_step,'logical_rows':done_rows,'processed_tokens':processed}
      save(out/'status.json',status);print(json.dumps(status),flush=True)
      wr.log({'train/loss':status['loss'],'progress/stage':stage,'progress/epoch':epoch+1,'progress/percent':100*done_rows/8400,'progress/optimizer_step':global_step,'train/elapsed_seconds':status['elapsed_seconds'],'train/tokens_per_second':processed/max(1,status['elapsed_seconds'])})
     del b,pred,v
    score=validate(model,tok,selection,stage,size)
    rec={'stage':stage,'epoch':epoch,'train_loss':running/n,'selection_loss':score,'epoch_seconds':time.time()-epochstart};history.append(rec);save(out/'history.json',history)
    wr.log({'epoch/index':len(history),'epoch/train_loss':rec['train_loss'],'epoch/validation_loss':score,'epoch/seconds':rec['epoch_seconds']})
    cp=out/f'stage{stage}-epoch{epoch}-adapters.safetensors';checkpoint(model,cp)
    if score<best:best=score;bestpath=cp;save(out/f'stage{stage}-selection.json',{**rec,'checkpoint':cp.name})
   restore(model,bestpath);del opt,sched;torch.cuda.empty_cache()
  save(out/'status.json',{'state':'trained_calibration_pending','elapsed_seconds':time.time()-started,'optimizer_steps':global_step,'processed_tokens':processed});wr.summary['training_state']='finished';wr.log({'progress/percent':100});wr.finish()
 except Exception as e:
  save(out/'status.json',{'state':'failed','error':repr(e),'elapsed_seconds':time.time()-started});wr.summary['training_state']='failed';wr.finish(exit_code=1);raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');args=a.parse_args()
 try:main(args.name)
 except Exception as e:
  p=ROOT/'runs'/args.name/'run';p.mkdir(parents=True,exist_ok=True);save(p/'status.json',{'state':'failed','error':repr(e)});traceback.print_exc();raise
