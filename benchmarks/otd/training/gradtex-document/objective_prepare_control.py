from pathlib import Path
import json,shutil,sys,hashlib
from objective_patch import patch
R=Path('/data/workspace/paper-diversity-v1');name='objective-document-control-v1';dst=R/name;src=R/'control'
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
assert not dst.exists(), 'Refuse to overwrite existing control'
dst.mkdir();(dst/'configs').mkdir();(dst/'prepared').mkdir()
for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
shutil.copy2(src/'models.lock.json',dst/'models.lock.json');(dst/'vendor').symlink_to(src/'vendor')
shutil.copy2(src/'configs/encoder.json',dst/'configs/encoder.json')
for p in (src/'prepared').glob('*.gz'):(dst/'prepared'/p.name).symlink_to(p)
m=json.loads((src/'prepared/manifest.json').read_text())
spec={'name':name,'source':'control','seed':42,'question':'Matched control for auxiliary document-level supervision of mixed edits','stage1':'unchanged segment loss; document head initialization preserves RNG','stage2':'original paper combined loss + 0.1 auxiliary mean-hidden document CE','document_label':'any AI involvement only if complete known region coverage','validation':'original combined paper selection loss, no document loss','evaluation':'existing calibrated token-mean document score and frozen suites; auxiliary head not deployed','data_order_and_tokens':'identical control','new_generation':False,'precision':'BF16 all model forwards'}
m['experiment']=spec;save(dst/'prepared/manifest.json',m);save(dst/'spec.json',spec)
patch(dst)
# Algebra/unit check on BF16 logits, no model inference; real BF16 preflight required by worker.
import os
os.chdir(dst);sys.path.insert(0,str(dst))
import torch
from modeling import training_loss,loss
z=torch.tensor([[[.1,.2],[.5,-.2]]],dtype=torch.bfloat16,requires_grad=True)
d=torch.tensor([[.3,-.4]],dtype=torch.bfloat16,requires_grad=True)
s=torch.zeros(1,15,dtype=torch.bfloat16,requires_grad=True);mix=torch.zeros(1,2,dtype=torch.bfloat16,requires_grad=True)
b={'labels':torch.full((1,2),-100),'target':torch.zeros(1,2,dtype=torch.bool),'segment_labels':torch.tensor([-100]),'mixed_labels':torch.tensor([-100]),'sentence_groups':[[]],'document_only':torch.tensor([True]),'document_labels':torch.tensor([1])}
y,parts=training_loss({'tokens':z,'document':d,'segment':s,'mixed':mix},b,2);y.backward()
assert z.grad is not None and not z.grad.any() and s.grad is None and mix.grad is None
assert d.grad.abs().sum()>0
save(dst/'objective-algebra-check.json',{'passed':True,'BF16_synthetic_logits':True,'document_only_token_gradient_zero':True,'document_gradient_nonzero':True,'no_model_inference':True})
save(dst/'DISPATCH_READY.json',{'protocol_validated':True,'GPU_preflight_required':True,'manifest_sha256':hashlib.sha256((dst/'prepared/manifest.json').read_bytes()).hexdigest(),'synthetic_gradient_check':True})
save(R/'auto-dispatch/jobs'/f'{name}.json',{'id':name,'rank':9,'enabled':True,'ready_file':str(dst/'DISPATCH_READY.json'),'completion_file':str(R/(name+'-status.json')),'dependencies':[],'command':[sys.executable,'-u',str(R/'worker.py'),name],'cwd':str(R)})
print('registered',name)
