"""Explicit two-stage training. Preparation/preflight never enters this entry point."""
from runtime import require_space, SPACE_CACHE
import argparse,gzip,hashlib,json,math,random,time,sys,platform
from pathlib import Path
import numpy as np
import bitsandbytes as bnb
import torch
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from safetensors.torch import save_file,load_file
from data import encode_example,sha
from modeling import Detector,collate,loss,training_loss
from adapters import attach_lora,parameter_groups
ROOT=Path(__file__).resolve().parent

def save(path,obj):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def read_data(name,manifest):
 blob=gzip.decompress((ROOT/'prepared-v2'/(name+'.jsonl.gz')).read_bytes())
 if hashlib.sha256(blob).hexdigest()!=manifest['files'][name]['sha256']:raise ValueError('Prepared data drift: '+name)
 return [json.loads(l) for l in blob.splitlines()]
def require_gpu():
 require_space()
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required; no FP32 fallback')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(8)
def configure_checkpointing(model,cfg,stage):
 enabled=cfg['stages'][str(stage)].get('gradient_checkpointing',True)
 if enabled:model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 else:model.backbone.gradient_checkpointing_disable()

def batches(rows,tok,kind,stage,size):
 for start in range(0,len(rows),size):yield collate([encode_example(r,tok,kind,stage) for r in rows[start:start+size]],tok.pad_token_id)
@torch.inference_mode()
def validation(model,tok,rows,stage,batch_size):
 model.eval();total=0.;n=0
 for batch in batches(rows,tok,model.kind,stage,batch_size):
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,_=loss(out,batch,stage)
  count=len(batch['labels']);total+=float(value)*count;n+=count
 return total/n

def main(a):
 require_gpu();cfg=json.loads((ROOT/'configs'/(a.flow+'.json')).read_text());info=json.loads((ROOT/'models.lock.json').read_text())[cfg.get('model_key',cfg['kind'])];manifest=json.loads((ROOT/'prepared-v2/manifest.json').read_text())
 if manifest['models'][cfg.get('model_key',cfg['kind'])]!=info:raise ValueError('Tokenizer/base revision mismatch')
 ready=json.loads((ROOT/'READY.json').read_text())
 if ready['status']!='prepared_not_trained' or ready['data_manifest_sha256']!=hashlib.sha256((ROOT/'prepared-v2/manifest.json').read_bytes()).hexdigest():raise ValueError('Preparation must be finalized before training')
 for name,digest in ready['source_code_sha256'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Prepared code changed: '+name)
 if hashlib.sha256((ROOT/'configs'/(a.flow+'.json')).read_bytes()).hexdigest()!=ready['configs_sha256'][a.flow+'.json']:raise ValueError('Prepared configuration changed')

 out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise RuntimeError('Use a new output directory; completed/partial runs are never overwritten')
 out.mkdir(parents=True,exist_ok=True)
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],trust_remote_code=False)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=Detector.load_base(info,cfg['kind']);coverage=attach_lora(model,cfg);model=model.cuda()
 import importlib.metadata as meta
 contract={'config':cfg,'model':info,'data_manifest_sha256':sha((ROOT/'prepared-v2/manifest.json').read_text()),'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'environment':{k:meta.version(k) for k in ['torch','transformers','numpy','safetensors','tokenizers','bitsandbytes','peft']},'gpu':torch.cuda.get_device_name(),'precision':'BF16 autocast; FP32 trainable masters; frozen linear storage per configuration; 8-bit Adam states (small tensors FP32)','parameters':sum(p.numel() for p in model.parameters()),'python':platform.python_version()}
 contract['kernel_environment']={k:meta.version(k) for k in ['fla-core','flash-linear-attention','einops','triton']} if cfg.get('fast_linear_attention') else {};contract['frozen_linear_dtype']=cfg.get('frozen_linear_dtype','float32');contract['adapter_coverage']=coverage;contract['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad);save(out/'run.json',contract);tok.save_pretrained(out/'tokenizer');model.backbone.config.save_pretrained(out/'backbone_config')
 selection=read_data('selection-windows',manifest);history=[];started=time.time();trained_source_tokens=0;processed_input_tokens=0
 for stage in [1,2]:
  best=float('inf');bestpath=out/f'stage{stage}-best.safetensors';epochs=cfg['stages'][str(stage)]['epochs'];lr=cfg['learning_rate'];batch_size=cfg['stages'][str(stage)].get('micro_batch',cfg['micro_batch']);accum=cfg['effective_batch']//batch_size
  configure_checkpointing(model,cfg,stage)
  if cfg['effective_batch']%batch_size:raise ValueError('Effective batch must divide micro batch')
  optimizer=bnb.optim.AdamW8bit(parameter_groups(model,cfg),lr=lr,weight_decay=.01)
  total_steps=sum(math.ceil(manifest['files'][f'stage{stage}-epoch{e}']['rows']/cfg['effective_batch']) for e in range(epochs));scheduler=get_cosine_schedule_with_warmup(optimizer,math.ceil(.06*total_steps),total_steps)
  for epoch in range(epochs):
   rows=read_data(f'stage{stage}-epoch{epoch}',manifest);model.train();optimizer.zero_grad(set_to_none=True);running=0.;steps=0
   n_batches=math.ceil(len(rows)/batch_size);epoch_start=time.time()
   for step,batch in enumerate(batches(rows,tok,cfg['kind'],stage,batch_size)):
    # Correct normalization for a partial final accumulation group.
    group_start=(step//accum)*accum;group_count=min(accum,n_batches-group_start)
    with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(batch);value,parts=training_loss(outputs,batch,stage)
    if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
    (value/group_count).backward();running+=float(value.detach());steps+=1
    trained_source_tokens+=int((batch['labels']>=0).sum());processed_input_tokens+=int(batch['attention_mask'].sum())
    if (step+1)%accum==0 or step+1==n_batches:
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    if step%50==0:
     status={'state':'training','stage':stage,'epoch':epoch,'batch':step,'batches':n_batches,'loss':float(value.detach()),'elapsed_seconds':time.time()-started};save(out/'status.json',status);print(json.dumps(status),flush=True)
   score=validation(model,tok,selection,stage,batch_size)
   record={'stage':stage,'epoch':epoch,'train_loss':running/steps,'selection_loss':score,'epoch_seconds':time.time()-epoch_start,'source_supervised_tokens_cumulative':trained_source_tokens,'processed_tokens_cumulative':processed_input_tokens};history.append(record);save(out/'history.json',history)
   if score<best:
    best=score;save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(bestpath));save(out/f'stage{stage}-selection.json',record)
  # Stage 2 starts from the selected stage-1 checkpoint with a fresh optimizer.
  model.load_state_dict(load_file(str(bestpath),device='cuda'));del optimizer,scheduler;torch.cuda.empty_cache()
 save(out/'status.json',{'state':'trained_calibration_pending','elapsed_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'source_supervised_tokens':trained_source_tokens,'processed_input_tokens':processed_input_tokens})
 print('Training finished. Run calibrate.py before evaluation; no test inference was performed.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--flow',choices=['encoder','causal','qwen35'],required=True);p.add_argument('--output',required=True);main(p.parse_args())
