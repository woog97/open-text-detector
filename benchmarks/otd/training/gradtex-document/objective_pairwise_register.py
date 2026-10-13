"""Register only after shared pair-data budget and algebra checks pass."""
from pathlib import Path
import json,sys,shutil,hashlib,subprocess
from objective_pairwise_patch import patch
R=Path('/data/workspace/paper-diversity-v1');O=R/'objective-pair-ranking';src=R/'control'
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
budget=json.loads((O/'budget-manifest.json').read_text());assert budget['state']=='data_prepared_not_gpu_ready'
for x in budget['checks'].values():assert .099<=x['paired_token_share']<=.101 and x['all_pairs_within_microbatch8'] and x['max_pair_draws_per_source']<=3 and x['all_adapter_rows_valid']
from transformers import AutoTokenizer
from objective_pairwise_validate import validate
tok=AutoTokenizer.from_pretrained(src/'run/tokenizer',local_files_only=True)
independent=validate(O,src,tok);save(O/'independent-publication-checks.json',independent)
for name in ['paper-ranking-control-v1','paper-ranking-v1']:
 assert not (R/name).exists(),'Existing arm: inspect before deliberate recovery'
 assert not (R/'auto-dispatch/jobs'/f'{name}.json').exists(),'Existing job: refuse duplicate'
r=subprocess.run([sys.executable,str(O/'objective_pairwise_check.py')],capture_output=True,text=True);assert r.returncode==0,r.stderr;save(O/'algebra-checks.json',json.loads(r.stdout))
publications=[]
for name,coef in [('paper-ranking-control-v1',0.),('paper-ranking-v1',.1)]:
 dst=R/name;assert not dst.exists(),'Refuse existing arm'
 dst.mkdir();(dst/'configs').mkdir();(dst/'prepared').mkdir()
 for p in src.glob('*.py'):shutil.copy2(p,dst/p.name)
 shutil.copy2(src/'models.lock.json',dst/'models.lock.json');(dst/'vendor').symlink_to(src/'vendor')
 cfg=json.loads((src/'configs/encoder.json').read_text());cfg['ranking_coefficient']=coef;assert cfg['effective_batch']==32 and all(cfg['stages'][str(stage)].get('micro_batch',cfg['micro_batch'])==8 for stage in [1,2]);save(dst/'configs/encoder.json',cfg)
 for p in (O/'prepared').glob('*.gz'):(dst/'prepared'/p.name).symlink_to(p)
 m=json.loads((O/'prepared/manifest.json').read_text());spec={'name':name,'seed':42,'ranking_coefficient':coef,'matched_control':'paper-ranking-control-v1','question':'Does matched generated/original ordering help localization and strict human FPR beyond absolute classification?','data':'identical pair-batched target crops for both arms; originalTRAIN verified provenance','classification':'original token+.2sentence+.2segment+.1mixed stage2, originalsegmentstage1','ranking':'coefficient * paired-row fraction * mean softplus(human-AI mean target tokenlogodds); bounded derivative, noextrahead','budget':budget['checks'],'selection':'unchanged combined paper validation loss excludesranking','evaluation':'calibration separate; existingworkflow/comparison; no testfitting','new_generation':False}
 m['experiment']=spec;save(dst/'prepared/manifest.json',m);save(dst/'spec.json',spec);patch(dst,coef)
 save(dst/'DISPATCH_READY.json',{'protocol_validated':True,'GPU_pair_preflight_required':True,'manifest_sha256':hashlib.sha256((dst/'prepared/manifest.json').read_bytes()).hexdigest(),'algebra_checks_passed':True})
 publications.append((R/'auto-dispatch/jobs'/f'{name}.json',{'id':name,'rank':9,'enabled':True,'ready_file':str(dst/'DISPATCH_READY.json'),'completion_file':str(R/(name+'-status.json')),'dependencies':[],'command':[sys.executable,'-u',str(R/'worker.py'),name],'cwd':str(R)}))
 audit=name+'-selection';publications.append((R/'auto-dispatch/jobs'/f'{audit}.json',{'id':audit,'rank':10,'enabled':True,'ready_file':str(dst/'DISPATCH_READY.json'),'completion_file':str(R/'auto-dispatch/audits'/name/'analysis-status.json'),'dependencies':[name],'command':[sys.executable,'-u',str(R/'auto-dispatch/selection_followup.py'),name],'cwd':str(R)}))
 print('prepared',name,'andselection')
for path,record in publications:
 assert not path.exists(),'Existing publication; deliberate recovery required'
 save(path,record)
 print('registered',record['id'])
