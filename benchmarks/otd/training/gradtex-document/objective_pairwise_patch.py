"""Isolated integration for verified paper pairs only; not HIP token supervision."""
from pathlib import Path
import shutil

def patch(root, coefficient):
    if coefficient not in (0.0,0.1):raise ValueError('Only matched0/.1pilot')
    root=Path(root)
    shutil.copy2(Path(__file__).with_name('objective_pairwise.py'),root/'pairwise.py')
    p=root/'data.py';s=p.read_text();assert '_pair_original_encode' not in s
    s+='''
_pair_original_encode=encode_example
def encode_example(row,tok,kind,stage):
 e=_pair_original_encode(row,tok,kind,stage)
 e['ranking_pair_id']=row.get('ranking_pair_id')
 e['ranking_role']=row.get('ranking_role')
 e['verified_pair_relation']=row.get('verified_pair_relation',False)
 e['ranking_mask']=[bool(t and row['text'][a:b].strip()) for t,(a,b) in zip(e['target'],e['offsets'])]
 if e['ranking_pair_id'] is not None:
  if stage!=2 or not e['verified_pair_relation']:raise ValueError('Ranking requires verified stage2pair')
  expected=0 if e['ranking_role']=='human' else 1
  if e['ranking_role'] not in ('human','ai') or any(y!=expected for y,m in zip(e['source_labels'],e['ranking_mask']) if m):raise ValueError('Ranking role conflicts with verified token provenance')
 return e
'''
    p.write_text(s)
    p=root/'modeling.py';s=p.read_text();assert '_pair_original_collate' not in s
    s+='''
from pairwise import pair_indices,ranking_loss
_pair_original_collate=collate
def collate(examples,pad_id,device='cuda'):
 b=_pair_original_collate(examples,pad_id,device);k=b['labels'].shape[1]
 b['ranking_pairs']=pair_indices(examples)
 b['ranking_mask']=torch.tensor([e.get('ranking_mask',[False]*len(e['source_labels']))+[False]*(k-len(e['source_labels'])) for e in examples],device=device,dtype=torch.bool)
 return b

def training_loss(outputs,batch,stage):
 base,parts=loss(outputs,batch,stage)
 if stage==1:
  if batch['ranking_pairs']:raise ValueError('No stage1ranking')
  return base,parts
 ranking=ranking_loss(outputs['tokens'],batch['ranking_mask'],batch['ranking_pairs'],RANKING_COEFFICIENT)
 parts['ranking']=float(ranking.detach())
 return base+ranking,parts
'''.replace('RANKING_COEFFICIENT',repr(coefficient))
    p.write_text(s)
    p=root/'train.py';s=p.read_text();assert 'value,parts=loss(outputs,batch,stage)' in s
    s=s.replace('from modeling import Detector,collate,loss','from modeling import Detector,collate,loss,training_loss').replace('value,parts=loss(outputs,batch,stage)','value,parts=training_loss(outputs,batch,stage)');p.write_text(s)
    # A pair-aware real GPU preflight must be added BEFORE any READY publication.
    for p in root.glob('*.py'):compile(p.read_text(),str(p),'exec')
    p=root/'preflight.py';s=p.read_text().replace('from modeling import Detector,collate,loss','from modeling import Detector,collate,loss,training_loss')
    needle="longest=max(candidates,key=lambda r:len(tok(r['text'],add_special_tokens=False)['input_ids']))"
    assert needle in s;s=s.replace(needle,needle+"\n  longest=dict(longest)\n  for key in ['ranking_pair_id','ranking_role','verified_pair_relation']:longest.pop(key,None)")
    needle='  # Round-trip trainable state';assert needle in s
    check='''  # Complete verified training pair; this runs before READY or optimizer steps.
  pairs={}
  for row in rows:
   if row.get('ranking_pair_id'):pairs.setdefault(row['ranking_pair_id'],[]).append(row)
  pair=max(pairs.values(),key=lambda rr:sum(len(r['text']) for r in rr))
  assert len(pair)==2
  pb=collate([encode_example(r,tok,cfg['kind'],2) for r in pair],tok.pad_token_id)
  assert len(pb['ranking_pairs'])==1
  model.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):po=model(pb);pv,pp=training_loss(po,pb,2)
  assert po['tokens'].dtype==torch.bfloat16 and torch.isfinite(pv)
  pv.backward()
  assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  assert model.token_head.weight.grad.abs().sum()>0
  assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.backbone.named_parameters() if '.lora_B.' in n)
  checks.append({'complete_pair_bf16':True,'finite_pair_loss_gradients':True,'ranking_term':pp['ranking'],'frozen_base_no_gradients':True})
  del pb,po,pv
'''
    s=s.replace(needle,check+needle);p.write_text(s)
    for p in root.glob('*.py'):compile(p.read_text(),str(p),'exec')
