from pathlib import Path
import json,gzip,hashlib,shutil,sys,collections
R=Path('/data/workspace/paper-diversity-v1');D=R/'auto-dispatch'
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def read(p):return [json.loads(l) for l in gzip.decompress(p.read_bytes()).splitlines()]
trials=[('boundary-seed73','boundary',73,5),('objective-token-sentence-v1','control',42,8),('mage10-v1','mage',42,14),('raid10-v1','raid',42,14)]
for name,source,seed,rank in trials:assert not (R/name).exists(),name
for name,source,seed,rank in trials:
 src=R/source;dst=R/name;dst.mkdir();(dst/'configs').mkdir();(dst/'prepared').mkdir()
 for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
 shutil.copy2(src/'models.lock.json',dst/'models.lock.json');(dst/'vendor').symlink_to(src/'vendor')
 cfg=json.loads((src/'configs/encoder.json').read_text());cfg['seed']=seed;save(dst/'configs/encoder.json',cfg)
 m=json.loads((src/'prepared/manifest.json').read_text());spec={'name':name,'source':source,'seed':seed,'rank':rank,'unchanged':'stages, updates, learning rates, validation-loss selection, BF16, original data split','generation':False}
 if name.endswith('10-v1'):
  from transformers import AutoTokenizer
  lock=json.loads((src/'models.lock.json').read_text());info=lock[cfg.get('model_key',cfg['kind'])];tok=AutoTokenizer.from_pretrained(R/'control/run/tokenizer',local_files_only=True)
  checks={}
  for p in (src/'prepared').glob('*.gz'):
   if not p.name.startswith('stage'):(dst/'prepared'/p.name).symlink_to(p);continue
   mixed=read(p);base=read(R/'control/prepared'/p.name);assert len(mixed)==len(base)
   lengths=[len(x)+2 for x in tok([x['text'] for x in mixed],add_special_tokens=False)['input_ids']]
   base_lengths=[len(x)+2 for x in tok([x['text'] for x in base],add_special_tokens=False)['input_ids']]
   # Deterministic prefix-spread thinning preserves retained donor relative order.
   candidates=[i for i,x in enumerate(mixed) if x.get('dataset')==source];total=sum(lengths);target=.10*total
   retained=set();exposure=0
   for parity in [0,1]:
    for i in candidates[parity::2]:
     if exposure>=target:break
     retained.add(i);exposure+=lengths[i]
   out=[x if i in retained else base[i] for i,x in enumerate(mixed)]
   outlengths=[lengths[i] if i in retained else base_lengths[i] for i in range(len(out))]
   share=exposure/sum(outlengths);assert .099<=share<=.101,(name,share)
   assert abs(sum(outlengths)/sum(base_lengths)-1)<.005
   # All retained external rows are from previously audited donors; paper rows from control.
   blob=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out).encode();(dst/'prepared'/p.name).write_bytes(gzip.compress(blob,mtime=0));key=p.name.removesuffix('.jsonl.gz');m['files'][key]={'rows':len(out),'sha256':hashlib.sha256(blob).hexdigest(),'papers':len({x['paper_id'] for x in out})}
   checks[key]={'external_token_share':share,'processed_tokens':sum(outlengths),'control_tokens':sum(base_lengths),'donor_draws':len(retained)}
  spec.update(question='Does reducing external exposure from20% to10% reduce paper regression?',checks=checks,selection='exploratory source-specific dose comparison, not a broad ratio sweep')
 else:
  for p in (src/'prepared').glob('*.gz'):(dst/'prepared'/p.name).symlink_to(p)
 if name=='boundary-seed73':spec.update(question='Does boundary sentence benefit replicate?',matched_control='control-seed73',data_order='identical boundary seed42 data; RNG seed changed only')
 if name=='objective-token-sentence-v1':
  p=dst/'modeling.py';s=p.read_text();s+='\n\ndef training_loss(outputs,batch,stage):\n total,parts=loss(outputs,batch,stage)\n if stage==1:return total,parts\n def ce(z,y):\n  m=y>=0\n  return F.cross_entropy(z.float()[m],y[m]) if m.any() else z.sum()*0\n return total-.2*ce(outputs["segment"],batch["segment_labels"])-.1*ce(outputs["mixed"],batch["mixed_labels"]),parts\n';p.write_text(s)
  p=dst/'train.py';s=p.read_text();assert 'value,parts=loss(outputs,batch,stage)' in s;s=s.replace('from modeling import Detector,collate,loss','from modeling import Detector,collate,loss,training_loss').replace('value,parts=loss(outputs,batch,stage)','value,parts=training_loss(outputs,batch,stage)');p.write_text(s)
  spec.update(question='Does sentence supervision recover the token-only regression without segment/mixed objectives?',stage1='unchanged',stage2='token + .2 sentence',selection_loss='original combined validation objective unchanged')
 m['experiment']=spec;save(dst/'prepared/manifest.json',m);save(dst/'spec.json',spec)
 for p in dst.glob('*.py'):compile(p.read_text(),str(p),'exec')
 # READY for dispatcher means validated launch protocol, GPU preflight is worker's first step.
 ready=dst/'DISPATCH_READY.json';save(ready,{'protocol_validated':True,'manifest_sha256':hashlib.sha256((dst/'prepared/manifest.json').read_bytes()).hexdigest(),'GPU_preflight':'required by worker before training','spec':spec})
 save(D/'jobs'/f'{name}.json',{'id':name,'rank':rank,'enabled':True,'ready_file':str(ready),'completion_file':str(R/(name+'-status.json')),'dependencies':[],'command':[sys.executable,'-u',str(R/'worker.py'),name],'cwd':str(R)})
 print('registered',name,rank)
