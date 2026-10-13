"""CPU/meta-only layer coverage and BF16 toy checkpoint/gradient checks; no model downloads."""
from runtime import require_space
import json,torch,copy
from pathlib import Path
from transformers import AutoConfig,AutoModel
from modeling import Detector
from adapters import attach_lora,parameter_groups
ROOT=Path(__file__).resolve().parent

def main():
 require_space();torch.set_num_threads(4);report={};lock=json.loads((ROOT/'models.lock.json').read_text())
 for key,info in lock.items():
  cfg=json.loads((ROOT/'configs'/(key+'.json')).read_text())
  conf=AutoConfig.from_pretrained(info['repo'],revision=info['revision'],local_files_only=True)
  if getattr(conf,'model_type',None)=='qwen3_5':conf=conf.text_config
  with torch.device('meta'):
   model=Detector(AutoModel.from_config(conf,attn_implementation='sdpa'),cfg['kind']);coverage=attach_lora(model,cfg)
  parameter_groups(model,cfg);report[key]=coverage
  assert any('.mlp.' in n for n in coverage['targets'])
  assert coverage['target_count']=={'encoder':112,'causal':196,'qwen35':248}[key]
  del model
 # Tiny random fixture: BF16 forward/backward, frozen base, and full-state round trip.
 class Backbone(torch.nn.Module):
  def __init__(self):
   super().__init__();self.config=type('Config',(),{'hidden_size':16})()
   self.mlp=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.GELU(),torch.nn.Linear(32,16))
  def forward(self,x):return self.mlp(x)
 cfg=json.loads((ROOT/'configs/encoder.json').read_text());cfg['lora']['rank']=4
 model=Detector(Backbone(),'encoder');attach_lora(model,cfg);groups=parameter_groups(model,cfg)
 frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
 opt=torch.optim.AdamW(groups);x=torch.randn(2,5,16)
 with torch.autocast('cpu',dtype=torch.bfloat16):out=model.token_head(model.backbone(x));loss=out.float().square().mean()
 assert out.dtype==torch.bfloat16;loss.backward();opt.step()
 assert all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen)
 assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.lora_B.' in n)
 clone=Detector(Backbone(),'encoder');attach_lora(clone,cfg);clone.load_state_dict(model.state_dict(),strict=True)
 with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
  assert torch.equal(model.token_head(model.backbone(x)),clone.token_head(clone.backbone(x)))
 report['toy_checks']={'bf16':True,'frozen_weights_unchanged':True,'adapter_gradients':True,'strict_full_state_roundtrip':True,'real_backbone_optimizer_steps':0}
 (ROOT/'coverage-audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:{'targets':v['target_count'],'trainable_backbone':v['backbone_trainable_parameters']} for k,v in report.items() if k!='toy_checks'}));print('BF16 toy checks passed')
if __name__=='__main__':main()
