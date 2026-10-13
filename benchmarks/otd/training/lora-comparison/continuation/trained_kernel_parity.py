"""Independent train-only numerical check at the completed full-tuned checkpoint."""
import sys,os,json,inspect,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRITON_CACHE_DIR']=str(HERE/'triton-cache');sys.path.insert(0,str(HERE/'kernel-vendor'));sys.path.insert(0,str(ROOT))
from train import require_gpu,read_data
from modeling import Detector,collate,loss
from data import encode_example
from transformers import AutoTokenizer
from safetensors.torch import load_file
import torch
require_gpu();cfg=json.loads((ROOT/'configs/qwen35.json').read_text());info=json.loads((ROOT/'models.lock.json').read_text())['qwen35'];manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
tok=AutoTokenizer.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
rows=read_data('stage2-epoch0',manifest)[512:544];examples=[encode_example(r,tok,cfg['kind'],2) for r in rows]
model=Detector.load_base(info,cfg['kind']);model.load_state_dict(load_file('/data/workspace/paper-backbone-comparison-v1/runs/qwen35/stage2-best.safetensors'));model.cuda().eval()
import transformers.models.qwen3_5.modeling_qwen3_5 as q
fast=q.torch_chunk_gated_delta_rule;reference=inspect.unwrap(fast)
assert reference is not fast and reference.__name__=='torch_chunk_gated_delta_rule'
values={}
with torch.inference_mode():
 for name,fn in [('reference',reference),('fast',fast)]:
  q.torch_chunk_gated_delta_rule=fn;probabilities=[];losses=[]
  for e in examples:
   batch=collate([e],tok.pad_token_id)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);value,_=loss(out,batch,2)
   assert all(v.dtype==torch.bfloat16 for v in out.values());probabilities.extend(out['tokens'].float().softmax(-1)[0,:,1].cpu().tolist());losses.append(float(value))
  values[name]={'probabilities':probabilities,'mean_loss':sum(losses)/len(losses)}
a=values['reference'];b=values['fast'];delta=[x-y for x,y in zip(a['probabilities'],b['probabilities'])]
rms=(sum(x*x for x in delta)/len(delta))**.5;relative=abs(b['mean_loss']/a['mean_loss']-1)
result={'source':'completed full-tuned Qwen35; independent training-only stage2 rows512:544','examples':32,'bf16_verified':True,'probability_rms_difference':rms,'probability_max_difference':max(abs(x) for x in delta),'reference_loss':a['mean_loss'],'fast_loss':b['mean_loss'],'loss_relative_difference':relative,'passed':rms<=.01 and relative<=.02}
(HERE/'trained-kernel-parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
