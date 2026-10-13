from pathlib import Path
import json,os,shutil,subprocess,sys,time,hashlib
R=Path('/data/workspace/paper-diversity-v1');out=R/'backlog-next';out.mkdir(exist_ok=True)
def save(p,x):p.write_text(json.dumps(x,indent=2))
# Refuse duplicates and preserve all active/completed files.
for name in ['analysis-process.json','ablation-process.json']:assert not (out/name).exists(),name
src=R/'control';dst=R/'objective-token-only-v1';assert not dst.exists()
gpus={int(a):(b.strip(),int(c)) for a,b,c in [x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]}
for gpu in [0,3]:assert gpus[gpu][1]<100,(gpu,gpus[gpu])
dst.mkdir();(dst/'configs').mkdir();(dst/'prepared').mkdir()
for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
for p in (src/'prepared').glob('*.gz'):(dst/'prepared'/p.name).symlink_to(p)
for p in ['models.lock.json','configs/encoder.json']:shutil.copy2(src/p,dst/p)
(dst/'vendor').symlink_to(src/'vendor')
m=json.loads((src/'prepared/manifest.json').read_text());m['experiment']={'arm':'objective-token-only-v1','change':'stage2 training loss token only; stage1 and validation loss unchanged','data':'identical frozen rows/order/tokens','selection':'original combined validation loss retained for comparability; objective-aware selection analysis separate'};save(dst/'prepared/manifest.json',m);save(dst/'spec.json',m['experiment'])
p=dst/'modeling.py';s=p.read_text();assert 'def loss(outputs,batch,stage):' in s
s+='\n\ndef training_loss(outputs,batch,stage):\n total,parts=loss(outputs,batch,stage)\n if stage==1:return total,parts\n # Same normalized token objective, remove auxiliary gradients only.\n import copy\n labels=batch["labels"];valid=labels>=0\n pertoken=F.cross_entropy(outputs["tokens"].float().flatten(0,1),labels.flatten(),ignore_index=-100,reduction="none").reshape(labels.shape)\n examples=[]\n for i in range(len(labels)):\n  target=valid[i]&batch["target"][i];context=valid[i]&~batch["target"][i]\n  terms=[pertoken[i][mask].mean() for mask in [target,context] if mask.any()]\n  if terms:examples.append(torch.stack(terms).mean())\n token=torch.stack(examples).mean() if examples else outputs["tokens"].sum()*0\n return token,parts\n'
p.write_text(s);p=dst/'train.py';s=p.read_text();assert 'from modeling import Detector,collate,loss' in s;s=s.replace('from modeling import Detector,collate,loss','from modeling import Detector,collate,loss,training_loss');assert 'value,parts=loss(outputs,batch,stage)' in s;s=s.replace('value,parts=loss(outputs,batch,stage)','value,parts=training_loss(outputs,batch,stage)');p.write_text(s)
# CPU gradient algebra check uses synthetic logits, no model forward or assets.
sys.path.insert(0,str(dst));import torch
from modeling import loss,training_loss
b={'labels':torch.tensor([[0,1]]),'target':torch.tensor([[False,True]]),'segment_labels':torch.tensor([1]),'mixed_labels':torch.tensor([1]),'sentence_groups':[[([0],0),([1],1)]]}
z={'tokens':torch.randn(1,2,2,requires_grad=True),'segment':torch.randn(1,15,requires_grad=True),'mixed':torch.randn(1,2,requires_grad=True)}
v,parts=training_loss(z,b,2);assert abs(v.item()-parts['token'])<1e-6;v.backward();assert z['tokens'].grad is not None and z['segment'].grad is None and z['mixed'].grad is None
save(out/'ablation-check.json',{'token_loss_equal':True,'auxiliary_gradients_absent':True,'stage1_unchanged':True,'selection_loss_unchanged':True})
for gpu,key,args in [(0,'analysis',[str(out/'analysis_mining.py')]),(3,'ablation',[str(R/'worker.py'),'objective-token-only-v1'])]:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpus[gpu][0],HF_HUB_CACHE='/data/workspace/model-cache',HF_HOME='/data/workspace/hf-home',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
 child=subprocess.Popen([sys.executable,'-u',*args],env=env,cwd=R,stdout=(out/(key+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 save(out/(key+'-process.json'),{'pid':child.pid,'gpu':gpu,'uuid':gpus[gpu][0],'time':time.time(),'args':args});print(key,child.pid,'GPU',gpu)
save(out/'plan.json',{'gpu0':['selection audit at calibration FPR .5/1/2% across six retained checkpoints','bounded existing-train-human mining','hard-human-v1 preflight/train/calibrate/evaluate'],'gpu3':['objective-token-only-v1 preflight/train/calibrate/evaluate'],'gpu1':'existing seed73 undisturbed','gpu2':'existing batch evaluation undisturbed','BF16':True,'no_generation':True})
