"""Run frozen evaluation profiles on idle Space GPUs; preserve outputs in Space bucket."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,shutil,traceback
from huggingface_hub import HfApi
R=Path('/tmp/pangram-eval-20261003');T=Path('/tmp/pangram-space-fast10');B='open-text-detector/training-storage';PREFIX='workspace/evaluations-20261003'
api=HfApi();R.mkdir(exist_ok=True)
def save(p,d):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def copy_tree(prefix,dest,accept=lambda p:True):
 jobs=[]
 for f in api.list_bucket_tree(B,prefix=prefix,recursive=True):
  if not hasattr(f,'size') or not f.path.startswith(prefix+'/'):continue
  rel=f.path[len(prefix)+1:]
  if not accept(rel):continue
  p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists() or p.stat().st_size!=f.size:jobs.append((f,p))
 if jobs:api.download_bucket_files(B,jobs,raise_on_missing_files=True)
def publish(p):api.batch_bucket_files(B,add=[(p,PREFIX+'/'+str(p.relative_to(R)))])
def prepare():
 save(R/'status.json',{'state':'copying existing Space files'})
 copy_tree('workspace/evaluation-suite-v1-20261003',R/'suite',lambda p:'__pycache__' not in p)
 copy_tree('workspace/current-data-v1/run',R/'modernbert-run',lambda p:p=='stage2-best.safetensors' or p.endswith('.json'))
 copy_tree('workspace/current-data-v1',R/'modernbert-code',lambda p:p in ['data.py','modeling.py','adapters.py','inference.py'])
 for name in ['modernbert','ettin']:
  d=R/(name+'-code');d.mkdir(exist_ok=True)
  (d/'runtime.py').write_text("from pathlib import Path\nimport sys,os\nROOT=Path(__file__).resolve().parent\nsys.path.insert(0,'/tmp/pangram-space-fast10/vendor')\nSPACE_CACHE='/tmp/pangram-space-fast10/assets'\ndef require_space():\n assert str(ROOT).startswith('/tmp/pangram-eval-20261003/') and Path('/tmp/pangram-space-fast10').is_dir()\nos.environ.setdefault('HF_HUB_OFFLINE','1')\n")
 modern=R/'modernbert-code/inference.py';s=modern.read_text().replace('return model.cuda().eval(),tok,contract','return model.to(dtype=torch.bfloat16,device="cuda").eval(),tok,contract');modern.write_text(s)
 d=R/'ettin-code'
 for f in ['data.py','modeling.py','adapters_short.py']:shutil.copy2(T/f,d/f)
 inference=s[s.index('def starts('):]
 (d/'inference.py').write_text('''from runtime import require_space
from pathlib import Path
import json,numpy as np,torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from data import layout
from modeling import Detector,collate
from adapters_short import attach_lora

def load_checkpoint(run):
 require_space();run=Path(run);contract=json.loads((run/'run.json').read_text());cfg=contract['config']
 model=Detector.load_base({'repo':contract['assets'],'revision':None},cfg['kind']);attach_lora(model,cfg)
 cp=json.loads((run/'stage2-selection.json').read_text())['checkpoint'];weights=load_file(str(run/cp))
 assert set(weights)=={n for n,p in model.named_parameters() if p.requires_grad}
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in weights:p.copy_(weights[n])
 tok=AutoTokenizer.from_pretrained(contract['assets'],local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model.physical_microbatch=8
 return model.to(dtype=torch.bfloat16,device='cuda').eval(),tok,contract

'''+inference)
 er=R/'ettin-run';er.mkdir(exist_ok=True);src=T/'runs/ettin-1b/run'
 cp=json.loads((src/'stage2-selection.json').read_text())['checkpoint']
 for f in ['run.json','stage2-selection.json',cp]:shutil.copy2(src/f,er/f)
 suite=R/'suite';m=json.loads((suite/'manifest.json').read_text());save(R/'original-suite-manifest.json',m)
 runner=suite/'run_eval.py';s=runner.read_text();s=s.replace("if not str(HERE).startswith('/data/workspace/'):","if not (str(HERE)=='/tmp/pangram-eval-20261003/suite' and Path('/tmp/pangram-space-fast10').is_dir()):")
 runner.write_text(s);m['files']['run_eval.py']=hashlib.sha256(runner.read_bytes()).hexdigest();m['runtime_change']='Space-local cache path allowed; data and scoring unchanged';save(suite/'manifest.json',m)
 save(R/'evaluation-notes.json',{'threshold':0.5,'calibrated':False,'modernbert_manuscripts':'Training overlaps later manuscript reservation; not an unseen test for ModernBERT.','reference':'Existing ModernBERT training tokenizer, shared across models','checkpoints':{'modernbert':'stage2-best.safetensors','ettin':cp},'precision':'BF16','training_runs_unmodified':True})
 for p in R.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.json'] and 'tokenizer' not in str(p):publish(p)

def launch(name,gpu,smoke):
 out=R/('smoke' if smoke else 'results')/name
 args=[sys.executable,'-u',str(R/'suite/run_eval.py'),'--run',str(R/(name+'-run')),'--adapter-dir',str(R/(name+'-code')),'--reference',str(R/'modernbert-run/tokenizer'),'--output',str(out),'--threshold','0.5','--batch-size','8','--rows-per-chunk','16']
 if smoke:args+=['--limit','2']
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(T/'vendor'),OMP_NUM_THREADS='8')
 log=R/(name+('-smoke' if smoke else '-eval')+'.log')
 with log.open('a') as f:p=subprocess.Popen(args,env=env,cwd=R,stdout=f,stderr=f,start_new_session=True)
 save(R/(name+'-process.json'),{'pid':p.pid,'gpu':gpu,'smoke':smoke,'output':str(out),'time':time.time()});return p
try:
 prepare();jobs={n:launch(n,g,True) for n,g in [('modernbert',0),('ettin',1)]};phase={n:'smoke' for n in jobs};uploaded={}
 while True:
  statuses={}
  for n,p in list(jobs.items()):
   rc=p.poll();path=R/('smoke' if phase[n]=='smoke' else 'results')/n/'status.json'
   statuses[n]={'phase':phase[n],'pid':p.pid,'returncode':rc,'progress':json.loads(path.read_text()) if path.exists() else None}
   if rc==0 and phase[n]=='smoke':
    assert statuses[n]['progress']['state']=='complete';jobs[n]=launch(n,0 if n=='modernbert' else 1,False);phase[n]='full';print('FULL EVAL STARTED',n,flush=True)
   elif rc is not None and rc!=0:phase[n]='failed'
  save(R/'status.json',statuses)
  for folder in [R/'smoke',R/'results']:
   if not folder.exists():continue
   for p in folder.rglob('*'):
    if not p.is_file() or p.suffix=='.tmp':continue
    stamp=(p.stat().st_size,p.stat().st_mtime_ns)
    if uploaded.get(str(p))==stamp:continue
    try:publish(p);uploaded[str(p)]=stamp
    except Exception as e:print('backup retry',type(e).__name__,flush=True)
  publish(R/'status.json')
  if all(p.poll() is not None for p in jobs.values()) and all(x!='smoke' for x in phase.values()):break
  time.sleep(15)
except Exception:
 save(R/'failure.json',{'error':traceback.format_exc()});traceback.print_exc();raise
