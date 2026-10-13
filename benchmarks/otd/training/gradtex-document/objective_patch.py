"""Isolated auxiliary document head; never promotes document labels to token gold."""
from pathlib import Path

DATA_APPEND = '''
_original_encode_example = encode_example
def encode_example(row,tok,kind,stage):
 if row.get('supervision')=='document_only':
  if stage!=2:raise ValueError('Document-only rows only allowed in stage2')
  if row.get('document_label') not in (0,1):raise ValueError('Binary document label required')
  if row.get('token_labels') is not None or row.get('regions'):raise ValueError('Document-only row cannot carry token gold')
  text=row['text'];e=tok(text,add_special_tokens=False,return_offsets_mapping=True)
  ids=e['input_ids'];off=e['offset_mapping']
  if not ids or len(ids)>510:raise ValueError('Whole document required; never truncate mixed document')
  out=layout(ids,[IGNORE]*len(ids),kind,stage,tok.cls_token_id,tok.sep_token_id)
  out.update(offsets=off,source_labels=[IGNORE]*len(ids),text=text,target=[False]*len(ids),segment_label=IGNORE,mixed_label=IGNORE,sentence_groups=[],document_label=row['document_label'])
 else:
  out=_original_encode_example(row,tok,kind,stage)
  regions=row['regions'];known=all(r['label'] in (0,1) for r in regions)
  coverage=sorted((r['start'],r['end']) for r in regions);end=0
  for a,b in coverage:
   if row['text'][end:a].strip():known=False
   end=max(end,b)
  if row['text'][end:].strip():known=False
  out['document_label']=int(any(r['label']==1 for r in regions)) if known else IGNORE
 out['document_only']=row.get('supervision')=='document_only'
 out['document_mask']=[bool(out['text'][a:b].strip()) for a,b in out['offsets']]
 if not any(out['document_mask']):raise ValueError('No document pooling tokens')
 return out
'''
MODEL_APPEND = '''
_original_collate=collate
_original_paper_loss=loss
def collate(examples,pad_id,device='cuda'):
 b=_original_collate(examples,pad_id,device);k=b['labels'].shape[1]
 b['document_labels']=torch.tensor([e['document_label'] for e in examples],device=device)
 b['document_only']=torch.tensor([e['document_only'] for e in examples],device=device,dtype=torch.bool)
 b['document_mask']=torch.tensor([e['document_mask']+[False]*(k-len(e['document_mask'])) for e in examples],device=device,dtype=torch.bool)
 return b

def training_loss(outputs,batch,stage):
 if stage==1:
  assert not batch['document_only'].any()
  return _original_paper_loss(outputs,batch,stage)
 doc=batch['document_only'];paper=~doc
 # External document targets never enter a token, sentence, segment, or mixed loss.
 assert (batch['labels'][doc]==-100).all()
 assert (batch['segment_labels'][doc]==-100).all() and (batch['mixed_labels'][doc]==-100).all()
 assert all(not batch['sentence_groups'][i] for i in torch.where(doc)[0].tolist())
 if paper.all():base,parts=_original_paper_loss(outputs,batch,stage)
 elif paper.any():
  ids=torch.where(paper)[0].tolist()
  sub={k:(v[paper] if isinstance(v,torch.Tensor) else [v[i] for i in ids]) for k,v in batch.items()}
  base,parts=_original_paper_loss({k:v[paper] for k,v in outputs.items()},sub,stage)
  base=base*paper.float().mean()
 else:base=outputs['tokens'].sum()*0;parts={}
 valid=batch['document_labels']>=0
 term=F.cross_entropy(outputs['document'].float()[valid],batch['document_labels'][valid]) if valid.any() else outputs['document'].sum()*0
 parts['document']=float(term.detach())
 return base+0.1*term,parts
'''

def patch(root):
 root=Path(root)
 p=root/'data.py';s=p.read_text();assert '_original_encode_example' not in s;p.write_text(s+DATA_APPEND)
 p=root/'modeling.py';s=p.read_text();assert '_original_paper_loss' not in s
 needle='self.mixed_head=nn.Linear(width,2)'
 assert needle in s;s=s.replace(needle,needle+'\n  with torch.random.fork_rng(devices=[]):self.document_head=nn.Linear(width,2)')
 needle="return {'tokens':self.token_head(source),'segment':self.segment_head(last),'mixed':self.mixed_head(last)}"
 assert needle in s;s=s.replace(needle,"mask=batch['document_mask'].unsqueeze(-1).to(source.dtype);pooled=(source*mask).sum(1)/mask.sum(1).clamp_min(1)\n  return {'tokens':self.token_head(source),'segment':self.segment_head(last),'mixed':self.mixed_head(last),'document':self.document_head(pooled)}")
 p.write_text(s+MODEL_APPEND)
 p=root/'adapters.py';s=p.read_text();s=s.replace('[model.token_head,model.segment_head,model.mixed_head]','[model.token_head,model.segment_head,model.mixed_head,model.document_head]');p.write_text(s)
 p=root/'train.py';s=p.read_text();s=s.replace('from modeling import Detector,collate,loss','from modeling import Detector,collate,loss,training_loss');assert 'value,parts=loss(outputs,batch,stage)' in s;s=s.replace('value,parts=loss(outputs,batch,stage)','value,parts=training_loss(outputs,batch,stage)');p.write_text(s)
 p=root/'inference.py';s=p.read_text();needle='sentence_groups=[]);examples.append(e)';assert needle in s;s=s.replace(needle,"sentence_groups=[],document_label=-100,document_only=False,document_mask=[bool(rows[i]['text'][c:d].strip()) for c,d in enc['offset_mapping'][i][a:b]]);examples.append(e)");p.write_text(s)
 for p in root.glob('*.py'):compile(p.read_text(),str(p),'exec')
 p=root/'preflight.py';s=p.read_text().replace('from modeling import Detector,collate,loss','from modeling import Detector,collate,loss,training_loss')
 s=s.replace('value,_=loss(out,batch,stage)','value,_=training_loss(out,batch,stage)')
 needle='  # Round-trip trainable state'
 assert needle in s
 check='''  # Real BF16 forward with document-only supervision: no token CE or auxiliary targets.
  docrow={'id':'preflight-document','text':longest['text'],'supervision':'document_only','document_label':1,'token_labels':None}
  db=collate([encode_example(docrow,tok,cfg['kind'],2)],tok.pad_token_id)
  model.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):do=model(db);dv,_=training_loss(do,db,2)
  assert do['document'].dtype==torch.bfloat16 and torch.isfinite(dv)
  dv.backward()
  assert model.document_head.weight.grad.abs().sum()>0
  assert all(p.grad is None or not p.grad.any() for h in [model.token_head,model.segment_head,model.mixed_head] for p in h.parameters())
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.backbone.named_parameters() if '.lora_B.' in n)
  checks.append({'document_only_bf16':True,'token_sentence_segment_mixed_head_gradients_zero':True,'document_head_and_adapter_gradients_nonzero':True})
  del db,do,dv
'''
 s=s.replace(needle,check+needle);p.write_text(s)
 for p in root.glob('*.py'):compile(p.read_text(),str(p),'exec')
