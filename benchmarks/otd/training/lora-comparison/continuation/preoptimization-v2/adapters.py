"""Explicit all-linear LoRA coverage; original projection names audited before injection."""
from runtime import require_space
from collections import Counter
import torch
from peft import LoraConfig,inject_adapter_in_model

def attach_lora(model,cfg):
 spec=cfg['lora'];backbone=model.backbone
 targets=[n for n,m in backbone.named_modules() if isinstance(m,torch.nn.Linear)]
 if not targets:raise ValueError('No linear projections found')
 if any('lora_' in n for n in targets):raise ValueError('Adapters already attached')
 original_other=[{'name':n,'shape':list(p.shape)} for n,p in backbone.named_parameters() if p.ndim>1 and not isinstance(backbone.get_submodule(n.rsplit('.',1)[0]),torch.nn.Linear)]
 config=LoraConfig(r=spec['rank'],lora_alpha=spec['alpha'],lora_dropout=spec['dropout'],bias='none',target_modules=targets,init_lora_weights=True,use_rslora=False)
 model.backbone=inject_adapter_in_model(config,backbone)
 found=[n for n,m in model.backbone.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A]
 if set(found)!=set(targets):raise AssertionError('Missing or unexpected LoRA projections')
 for name,p in model.backbone.named_parameters():
  expected='.lora_A.' in name or '.lora_B.' in name
  if p.requires_grad!=expected:raise AssertionError('Unexpected trainability: '+name)
 for head in [model.token_head,model.segment_head,model.mixed_head]:head.requires_grad_(True)
 return {'rank':spec['rank'],'alpha':spec['alpha'],'scaling':spec['alpha']/spec['rank'],'target_count':len(targets),'targets':targets,'target_families':dict(Counter(n.split('.')[-1] for n in targets)),'frozen_other_multidimensional_weights':original_other,'frozen_biases_and_norms':True,'heads':'fully trained','backbone_trainable_parameters':sum(p.numel() for p in backbone.parameters() if p.requires_grad),'all_original_linear_projections_covered':True}

def parameter_groups(model,cfg):
 adapters=[p for p in model.backbone.parameters() if p.requires_grad]
 heads=[p for head in [model.token_head,model.segment_head,model.mixed_head] for p in head.parameters() if p.requires_grad]
 assert len({id(p) for p in adapters+heads})==len(adapters+heads)
 assert {id(p) for p in adapters+heads}=={id(p) for p in model.parameters() if p.requires_grad}
 return [{'params':adapters,'lr':cfg['learning_rate']},{'params':heads,'lr':cfg['head_learning_rate']}]
