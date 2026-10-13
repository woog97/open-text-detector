from runtime import require_space
from collections import Counter
import torch
from peft import LoraConfig,inject_adapter_in_model

def attach_lora(model,cfg):
 require_space();spec=cfg['lora'];backbone=model.backbone
 targets=[n for n,m in backbone.named_modules() if isinstance(m,torch.nn.Linear)]
 # Fused MoE experts are Parameters, not Linear modules. Include both expert projections.
 params=[n for n,p in backbone.named_parameters() if p.ndim==3 and (n.endswith('gate_up_proj') or n.endswith('down_proj'))]
 if getattr(backbone.config,'num_experts',0):assert params,'MoE expert adapter targets missing'
 rank_pattern={key:spec.get('expert_rank',spec['rank']) for n in params for key in [n,n.rsplit('.',1)[0]]}
 config=LoraConfig(rank_pattern=rank_pattern,r=spec['rank'],lora_alpha=spec['alpha'],lora_dropout=0,bias='none',target_modules=targets,target_parameters=params or None,init_lora_weights=True,use_rslora=False)
 model.backbone=inject_adapter_in_model(config,backbone)
 rank_counts=Counter()
 for n,m in model.backbone.named_modules():
  if hasattr(m,'r') and isinstance(m.r,dict) and 'default' in m.r:
   expert=hasattr(m,'parameter_name') and m.parameter_name in ('gate_up_proj','down_proj')
   expected=spec.get('expert_rank',spec['rank']) if expert else spec['rank']
   assert m.r['default']==expected,(n,m.r,expected)
   rank_counts[('expert' if expert else 'non_expert')+':'+str(expected)]+=1
 assert rank_counts.get('expert:'+str(spec.get('expert_rank',spec['rank'])),0)==len(params)
 for n,p in model.backbone.named_parameters():
  if p.requires_grad and not ('lora_A' in n or 'lora_B' in n):raise AssertionError('Unexpected trainable base weight '+n)
  if p.requires_grad:p.data=p.data.to(torch.bfloat16)
 for head in [model.token_head,model.segment_head,model.mixed_head,model.document_head]:head.to(dtype=torch.bfloat16).requires_grad_(True)
 return {'actual_rank_counts':dict(rank_counts),'trainable_dtype':'bfloat16','linear_targets':targets,'expert_parameter_targets':params,'rank':spec['rank'],'expert_rank':spec.get('expert_rank',spec['rank']),'alpha':spec['alpha'],'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}

def parameter_groups(model,cfg):
 return [{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':cfg['learning_rate']},{'params':[p for h in [model.token_head,model.segment_head,model.mixed_head,model.document_head] for p in h.parameters()],'lr':cfg['head_learning_rate']}]
