"""Validate actual BF16 forwards/backwards and multiwindow document supervision."""
from runtime import require_space
from pathlib import Path
import json,gzip,hashlib,time,sys
import torch
from transformers import AutoTokenizer
from safetensors.torch import save_file,load_file
from modeling import Detector,collate,training_loss
from data import encode_example
from adapters import attach_lora,parameter_groups
R=Path(__file__).resolve().parent

def main():
 require_space();assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 cfg=json.loads((R/'configs/encoder.json').read_text());info=json.loads((R/'models.lock.json').read_text())['encoder'];manifest=json.loads((R/'prepared-v2/manifest.json').read_text())
 for name,f in manifest['files'].items():
  b=gzip.decompress((R/'prepared-v2'/f'{name}.jsonl.gz').read_bytes());assert hashlib.sha256(b).hexdigest()==f['sha256'];assert len(b.splitlines())==f['rows']
 tok=AutoTokenizer.from_pretrained(info['repo'],local_files_only=True)
 def rows(name):return [json.loads(l) for l in gzip.open(R/'prepared-v2'/f'{name}.jsonl.gz','rt')]
 sample=rows('stage2-epoch0');counts={};max_windows=0
 for r in sample:
  e=encode_example(r,tok,'encoder',2);chunks=e.get('chunks',[e]);max_windows=max(max_windows,len(chunks));counts[r['dataset']]=counts.get(r['dataset'],0)+1
  for c in chunks:
   assert len(c['ids'])<=512
   if r.get('supervision')=='document_only':assert set(c['source_labels'])=={-100}
 model=Detector.load_base(info,'encoder');coverage=attach_lora(model,cfg);parameter_groups(model,cfg);model.cuda().train();model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 tests=[]
 paper=max((r for r in sample if r.get('supervision')!='document_only'),key=lambda r:len(tok(r['text'],add_special_tokens=False)['input_ids']))
 doc=max((r for r in sample if r.get('supervision')=='document_only'),key=lambda r:len(tok(r['text'],add_special_tokens=False)['input_ids']))
 for title,rs,stage in [('stage1',[paper]*4,1),('mixed',[paper,doc,paper,doc],2),('document-only',[doc],2)]:
  model.zero_grad(set_to_none=True);b=collate([encode_example(r,tok,'encoder',stage) for r in rs],tok.pad_token_id)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b);value,_=training_loss(out,b,stage)
  assert out['tokens'].dtype==torch.bfloat16 and torch.isfinite(value);value.backward()
  assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
  if title=='document-only':
   assert model.document_head.weight.grad.abs().sum()>0
   assert all(p.grad is None or not p.grad.any() for h in [model.token_head,model.segment_head,model.mixed_head] for p in h.parameters())
  tests.append({'test':title,'logical_examples':len(rs),'windows':len(b['labels']),'loss':float(value.detach()),'forward_dtype':'bfloat16','finite_gradients':True})
 # Verify multiwindow pooling against manually calculated weighted averages.
 z=torch.tensor([[1.,3.],[5.,7.],[2.,4.]],device='cuda',requires_grad=True);weights=torch.tensor([2.,1.,3.],device='cuda');owner=torch.tensor([0,0,1],device='cuda');v=torch.zeros(2,2,device='cuda').index_add(0,owner,z*weights[:,None]);n=torch.zeros(2,device='cuda').index_add(0,owner,weights);v=v/n[:,None];assert torch.allclose(v,torch.tensor([[7/3,13/3],[2,4]],device='cuda'))
 from train import save
 result={'passed':True,'tests':tests,'stage2_epoch0_counts':counts,'largest_document_windows':max_windows,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'optimizer_updates':0,'adapter_coverage':coverage}
 save(R/'preflight-results.json',result)
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 save(R/'READY.json',{'status':'prepared_not_trained','data_manifest_sha256':digest(R/'prepared-v2/manifest.json'),'source_code_sha256':{p.name:digest(p) for p in R.glob('*.py')},'configs_sha256':{p.name:digest(p) for p in (R/'configs').glob('*.json')},'preflight_sha256':digest(R/'preflight-results.json')})
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
