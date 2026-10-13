from pathlib import Path
import json,gzip,hashlib,shutil,sys,os,collections
R=Path('/data/workspace/paper-diversity-v1');name='gradtex-document10-v1';dst=R/name;src=R/'objective-document-control-v1';data=R/'gradtex-document-preparation'
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
assert not dst.exists(),'Refuse overwrite'
checks=json.loads((data/'budget-manifest.json').read_text());assert checks['state']=='data_prepared'
for c in checks['checks'].values():
 assert .099<=c['external_token_share']<=.101 and c['rows']==12000 and c['max_draws_per_source']<=3
 assert abs(c['processed_tokens']/c['control_tokens']-1)<.005
 assert abs(c['class_tokens']['0']-c['class_tokens']['1'])/sum(c['class_tokens'].values())<.01
dst.mkdir();(dst/'configs').mkdir();(dst/'prepared').mkdir()
for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
shutil.copy2(src/'models.lock.json',dst/'models.lock.json');shutil.copy2(src/'configs/encoder.json',dst/'configs/encoder.json');(dst/'vendor').symlink_to(R/'control/vendor')
for p in (data/'prepared').glob('*.gz'):(dst/'prepared'/p.name).symlink_to(p)
# Stage1 preflight uses an original paper row, and document-only test uses a real donor.
p=dst/'preflight.py';s=p.read_text().replace("candidates=sorted(rows,key=lambda r:len(r['text']),reverse=True)[:1000]","candidates=sorted([r for r in rows if r.get('supervision')!='document_only'],key=lambda r:len(r['text']),reverse=True)[:1000]")
s=s.replace("docrow={'id':'preflight-document','text':longest['text'],'supervision':'document_only','document_label':1,'token_labels':None}","docrow=max([r for r in rows if r.get('supervision')=='document_only'],key=lambda r:len(r['text']))")
p.write_text(s)
manifest=json.loads((data/'prepared/manifest.json').read_text());spec=json.loads((src/'spec.json').read_text());spec.update(name=name,matched_control=src.name,data_order_and_tokens='control stage1;10%stage2 processed-token replacement; fixed drawcount and near-identical tokenbudget',donor_supervision='whole document only, AI-involved vs human; no localization targets',data_checks=checks['checks'],donor_source=str(data),control_loss='retained paper loss weighted by paper-row fraction, plus0.1documentCE on all known documents')
manifest['experiment']=spec;save(dst/'prepared/manifest.json',manifest);save(dst/'spec.json',spec)
for p in dst.glob('*.py'):compile(p.read_text(),str(p),'exec')
os.chdir(dst);sys.path.insert(0,str(dst))
from transformers import AutoTokenizer
from data import encode_example
from modeling import collate,training_loss
import torch
tok=AutoTokenizer.from_pretrained(R/'control/run/tokenizer',local_files_only=True)
counts={};donor=None
for key in ['stage1-epoch0','stage2-epoch0','stage2-epoch1','stage2-epoch2']:
 blob=gzip.decompress((dst/'prepared'/f'{key}.jsonl.gz').read_bytes());assert hashlib.sha256(blob).hexdigest()==manifest['files'][key]['sha256']
 rows=[json.loads(x) for x in blob.splitlines()];stage=1 if key.startswith('stage1') else 2;n=0
 for r in rows:
  e=encode_example(r,tok,'encoder',stage)
  if r.get('supervision')=='document_only':
   assert set(e['source_labels'])=={-100} and e['segment_label']==-100 and e['mixed_label']==-100 and not e['sentence_groups']
   assert len(e['ids'])<=512 and e['text']==r['text'];n+=1;donor=e
 counts[key]={'rows':len(rows),'document_only':n,'all_adapter_checks':True}
assert donor
b=collate([donor],tok.pad_token_id,device='cpu');out={k:torch.zeros(*shape,dtype=torch.bfloat16,requires_grad=True) for k,shape in {'tokens':(1,len(donor['source_labels']),2),'segment':(1,15),'mixed':(1,2),'document':(1,2)}.items()}
v,_=training_loss(out,b,2);v.backward();assert not out['tokens'].grad.any() and out['document'].grad.abs().sum()>0
save(dst/'adapter-validation.json',{'passed':True,'counts':counts,'document_only_token_ce_gradient_zero':True,'whole_document_preserved':True})
save(dst/'DISPATCH_READY.json',{'protocol_validated':True,'GPU_preflight_required':True,'manifest_sha256':hashlib.sha256((dst/'prepared/manifest.json').read_bytes()).hexdigest(),'adapter_validation':True})
save(R/'auto-dispatch/jobs'/f'{name}.json',{'id':name,'rank':9,'enabled':True,'ready_file':str(dst/'DISPATCH_READY.json'),'completion_file':str(R/(name+'-status.json')),'dependencies':[],'command':[sys.executable,'-u',str(R/'worker.py'),name],'cwd':str(R)})
print('registered',name,counts)
