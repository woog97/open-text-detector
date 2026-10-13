"""Remote-only preparation; source strings supplied by transport wrapper."""
from pathlib import Path
import json,hashlib,shutil,py_compile,sys,time
ROOT=Path('/data/workspace/paper-batch-sweep-v1/throughput-followup-v1')
BASE=Path('/data/workspace/paper-batch-sweep-v1/batch32-lr2e4')
D=Path('/data/workspace/paper-diversity-v1/auto-dispatch')
gate=ROOT/'mixed-gradient'
assert json.loads((gate/'status.json').read_text())['state']=='complete'
cases=json.loads((gate/'results.json').read_text())['cases'];assert len(cases)==4
for c in cases:
 assert c['bf16_verified'] and c['finite_gradients'] and c['frozen_verified']
 if 'gradient_cosine' in c:assert c['gradient_cosine']>.98 and c['gradient_relative_l2']<.2
 assert c['peak_reserved_bytes']<.7*c['gpu_total_bytes']
trials=[{'name':'micro16-8-lr1p4e4','lr':.00014},{'name':'micro16-8-lr2e4','lr':.0002},{'name':'micro16-8-lr2p8e4','lr':.00028}]
protocol={'trials':trials,'training_microbatches':{'1':16,'2':8},'checkpointing':False,'effective_batch':32,'validation_microbatch':4,'seed':42,'data':'Exact unchanged original batch32 control manifest and row order; 42k draws','selection':'Minimum stage2 validation loss before one winner frozen evaluation','intent':'Bounded time-to-validation-quality comparison, not claimed numerical equivalence','lr_rationale':'Geometric neighborhood of existing batch32 control LR; head LR remains 0.1 adapterLR. Existing batch64/128 losses cannot tune effective32 recipe.','precision':'BF16 forwards; original model storage unchanged','warm_start':False,'diagnostic_cases':cases,'baseline':str(BASE)}
assert not (ROOT/'PROTOCOL.json').exists(),'Existing preparation; inspect, do not overwrite'
(ROOT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
for name,source in SOURCES.items():
 p=ROOT/name;p.write_text(source);py_compile.compile(str(p),doraise=True)
for trial in trials:
 r=ROOT/trial['name'];assert not r.exists();r.mkdir();(r/'configs').mkdir();(r/'prepared').mkdir()
 for p in BASE.glob('*.py'):shutil.copy2(p,r/p.name)
 for p in (BASE/'configs').glob('*.json'):shutil.copy2(p,r/'configs'/p.name)
 shutil.copy2(BASE/'models.lock.json',r/'models.lock.json');(r/'vendor').symlink_to(BASE/'vendor',target_is_directory=True)
 for p in (BASE/'prepared').iterdir():
  if p.is_file():(r/'prepared'/p.name).symlink_to(p)
 cfg=json.loads((r/'configs/causal.json').read_text());cfg['learning_rate']=trial['lr'];cfg['head_learning_rate']=trial['lr']/10
 cfg['stages']['1'].update(micro_batch=16,gradient_checkpointing=False);cfg['stages']['2'].update(micro_batch=8,gradient_checkpointing=False)
 (r/'configs/causal.json').write_text(json.dumps(cfg,indent=2))
 p=r/'train.py';s=p.read_text();old='score=validation(model,tok,selection,stage,batch_size)';assert s.count(old)==1;s=s.replace(old,'score=validation(model,tok,selection,stage,4)');p.write_text(s);py_compile.compile(str(p),doraise=True)
 assert (r/'prepared/manifest.json').read_bytes()==(BASE/'prepared/manifest.json').read_bytes()
 (r/'spec.json').write_text(json.dumps({'trial':trial,'protocol':str(ROOT/'PROTOCOL.json'),'code_sha256':hashlib.sha256(s.encode()).hexdigest()},indent=2))
 (r/'DISPATCH_READY').write_text('Manifest exact; isolated config; fixed validation batch4; source compiles; mixed and long-example BF16 diagnostic gates passed. Worker performs production preflight before fresh training.')

def register(j):
 p=D/'jobs'/(j['id']+'.json');assert not p.exists();t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
for i,t in enumerate(trials):
 r=ROOT/t['name'];register({'id':'throughput-'+t['name'],'rank':8+i,'enabled':True,'ready_file':str(r/'DISPATCH_READY'),'completion_file':str(r/'status.json'),'dependencies':['throughput-mixed-gradient-v1'],'command':[sys.executable,'-u',str(ROOT/'worker.py'),t['name']],'cwd':str(ROOT)})
(ROOT/'FINALIZE_READY').write_text('Select all three by unchanged validation; calibrate and evaluate only winner, preserve prior frozen control.')
register({'id':'throughput-quality-comparison-v1','rank':14,'enabled':True,'ready_file':str(ROOT/'FINALIZE_READY'),'completion_file':str(ROOT/'comparison/status.json'),'dependencies':['throughput-'+t['name'] for t in trials],'command':[sys.executable,'-u',str(ROOT/'finalize.py')],'cwd':str(ROOT)})
print(json.dumps({'registered':trials,'finalizer':'throughput-quality-comparison-v1','gate':cases}))
