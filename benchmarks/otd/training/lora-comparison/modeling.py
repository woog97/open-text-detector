"""Shared supervised heads; encoder single-copy and causal Repeat2 backbones."""
from runtime import require_space, SPACE_CACHE
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

class Detector(nn.Module):
 def __init__(self,backbone,kind):
  super().__init__();self.backbone=backbone;self.kind=kind
  width=backbone.config.hidden_size
  self.token_head=nn.Linear(width,2);self.segment_head=nn.Linear(width,15);self.mixed_head=nn.Linear(width,2)
  if hasattr(backbone.config,'use_cache'):backbone.config.use_cache=False
 @classmethod
 def load_base(cls,info,kind):
  # FP32 master parameters with BF16 autocast for every forward; never FP32 inference.
  require_space()
  loaded=AutoModel.from_pretrained(info['repo'],revision=info['revision'],attn_implementation='sdpa',trust_remote_code=False,cache_dir=SPACE_CACHE,dtype=torch.float32)
  backbone=loaded.language_model if getattr(loaded.config,'model_type',None)=='qwen3_5' else loaded
  return cls(backbone,kind)
 def forward(self,batch):
  h=self.backbone(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
  positions=batch['positions'];source=h.gather(1,positions.unsqueeze(-1).expand(-1,-1,h.shape[-1]))
  last=h[torch.arange(h.shape[0],device=h.device),batch['last_positions']]
  return {'tokens':self.token_head(source),'segment':self.segment_head(last),'mixed':self.mixed_head(last)}

def collate(examples,pad_id,device='cuda'):
 maxseq=max(len(e['ids']) for e in examples);n=((maxseq+7)//8)*8;k=max(len(e['source_labels']) for e in examples)
 values={key:[] for key in ['input_ids','attention_mask','positions','labels','target','last_positions','segment_labels','mixed_labels']}
 for e in examples:
  length=len(e['ids']);s=len(e['source_labels'])
  values['input_ids'].append(e['ids']+[pad_id]*(n-length));values['attention_mask'].append([1]*length+[0]*(n-length))
  values['positions'].append(e['source_positions']+[0]*(k-s));values['labels'].append(e['source_labels']+[-100]*(k-s));values['target'].append(e['target']+[False]*(k-s));values['last_positions'].append(e['last_position']);values['segment_labels'].append(e['segment_label']);values['mixed_labels'].append(e['mixed_label'])
 batch={key:torch.tensor(value,device=device,dtype=torch.bool if key=='target' else torch.long) for key,value in values.items()};batch['sentence_groups']=[e['sentence_groups'] for e in examples];return batch

def loss(outputs,batch,stage):
 logits=outputs['tokens'].float();labels=batch['labels'];valid=labels>=0
 def masked_ce(logits,labels):
  mask=labels>=0
  return F.cross_entropy(logits.float()[mask],labels[mask]) if mask.any() else logits.sum()*0
 segment=masked_ce(outputs['segment'],batch['segment_labels'])
 if stage==1:return segment,{'segment':float(segment.detach())}
 pertoken=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').reshape(labels.shape)
 # Each example contributes equally; target/context each get half when both exist.
 examples=[]
 for i in range(len(labels)):
  target=valid[i]&batch['target'][i];context=valid[i]&~batch['target'][i];parts=[pertoken[i][mask].mean() for mask in [target,context] if mask.any()]
  if parts:examples.append(torch.stack(parts).mean())
 token=torch.stack(examples).mean() if examples else logits.sum()*0
 sentence_losses=[]
 for i,groups in enumerate(batch['sentence_groups']):
  terms=[F.cross_entropy(logits[i,indices].mean(0,keepdim=True),torch.tensor([label],device=logits.device)) for indices,label in groups]
  if terms:sentence_losses.append(torch.stack(terms).mean())
 sentence=torch.stack(sentence_losses).mean() if sentence_losses else logits.sum()*0
 mixed=masked_ce(outputs['mixed'],batch['mixed_labels']);total=token+.2*sentence+.2*segment+.1*mixed
 return total,{'token':float(token.detach()),'sentence':float(sentence.detach()),'segment':float(segment.detach()),'mixed':float(mixed.detach())}
