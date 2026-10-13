"""BF16 inference preserves Repeat2 at deployment and aligns only second-copy scores."""
from runtime import require_space, SPACE_CACHE
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoModel,AutoTokenizer
from transformers.initialization import no_init_weights
from safetensors.torch import load
from data import layout
from modeling import Detector,collate
from adapters import attach_lora

def load_checkpoint(run):
 require_space()
 run=Path(run);contract=json.loads((run/'run.json').read_text());kind=contract['config']['kind'];cfg=AutoConfig.from_pretrained(run/'backbone_config',local_files_only=True)
 with no_init_weights():model=Detector(AutoModel.from_config(cfg,attn_implementation='sdpa'),kind)
 attach_lora(model,contract['config'])
 model.load_state_dict(load((run/'stage2-best.safetensors').read_bytes()),strict=True)
 tok=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True);return model.cuda().eval(),tok,contract

def starts(n,width=510,stride=256):return [0] if n<=width else sorted(set(list(range(0,n-width+1,stride))+[n-width]))
@torch.inference_mode()
def predict(rows,model,tok,batch_size=8):
 enc=tok([r['text'] for r in rows],add_special_tokens=False,return_offsets_mapping=True,truncation=False);jobs=[];sums=[np.zeros(len(x),np.float64) for x in enc['input_ids']];den=[np.zeros(len(x),np.int32) for x in enc['input_ids']]
 for i,ids in enumerate(enc['input_ids']):
  if not ids:raise ValueError('Empty text cannot be evaluated')
  for start in starts(len(ids)):jobs.append((i,start,min(start+510,len(ids))))
 for pos in range(0,len(jobs),batch_size):
  part=jobs[pos:pos+batch_size];examples=[]
  for i,a,b in part:
   e=layout(enc['input_ids'][i][a:b],[-100]*(b-a),model.kind,2,tok.cls_token_id,tok.sep_token_id);e.update(source_labels=[-100]*(b-a),target=[False]*(b-a),segment_label=-100,mixed_label=-100,sentence_groups=[],document_label=-100,document_only=False,document_mask=[bool(rows[i]['text'][c:d].strip()) for c,d in enc['offset_mapping'][i][a:b]]);examples.append(e)
  batch=collate(examples,tok.pad_token_id)
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch)['tokens']
  probs=logits.float().softmax(-1)[...,1].cpu().numpy()
  for (i,a,b),p in zip(part,probs):sums[i][a:b]+=p[:b-a];den[i][a:b]+=1
 if not all((d>0).all() for d in den):raise AssertionError('Uncovered source token')
 return [(off,s/d) for off,s,d in zip(enc['offset_mapping'],sums,den)]
