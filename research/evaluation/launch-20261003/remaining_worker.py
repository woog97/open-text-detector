"""Evaluate one completed classifier; BF16, frozen suite, persistent bucket backup."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,shutil,threading,traceback
from huggingface_hub import HfApi
name=sys.argv[1];gpu=sys.argv[2];h200=name in ['gemma4-12b','qwen36-35b-a3b']
T=Path('/workspace/woog/pangram/backbones-20261003' if h200 else '/tmp/pangram-space-fast10')
R=(T/'evaluations-20261003' if h200 else Path('/tmp/pangram-eval-20261003/remaining'))/name
R.mkdir(parents=True,exist_ok=True);api=HfApi();B='open-text-detector/training-storage';PREFIX='workspace/evaluations-20261003/remaining/'+name

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def download(prefix,dest,accept=lambda x:True):
 jobs=[]
 for f in api.list_bucket_tree(B,prefix=prefix,recursive=True):
  if not hasattr(f,'size') or not f.path.startswith(prefix+'/'):continue
  rel=f.path[len(prefix)+1:];p=dest/rel
  if not accept(rel):continue
  p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists() or p.stat().st_size!=f.size:jobs.append((f,p))
 if jobs:api.download_bucket_files(B,jobs,raise_on_missing_files=True)

def backup():
 seen={}
 while True:
  try:
   files=[]
   for p in R.rglob('*'):
    if not p.is_file() or p.suffix in ['.tmp','.safetensors','.pyc'] or '__pycache__' in str(p) or 'suite/' in str(p) or 'reference/' in str(p):continue
    stamp=(p.stat().st_size,p.stat().st_mtime_ns)
    if seen.get(str(p))!=stamp:files.append((p,stamp))
   # Batch requests; prioritize final reports and statuses before raw chunks.
   files.sort(key=lambda x:('results.json' not in x[0].name,'status' not in x[0].name))
   for start in range(0,len(files),50):
    batch=files[start:start+50];api.batch_bucket_files(B,add=[(p,PREFIX+'/'+str(p.relative_to(R))) for p,_ in batch])
    for p,stamp in batch:seen[str(p)]=stamp
   save(R/'backup-status.json',{'files':len(seen),'time':time.time()})
   if (R/'finished.json').exists():return
  except Exception as e:print('backup retry',type(e).__name__,flush=True)
  time.sleep(30)
try:
 assert not (R/'worker-start.json').exists(),'Already launched; inspect before resuming'
 save(R/'worker-start.json',{'name':name,'gpu':gpu,'pid':os.getpid(),'time':time.time()})
 download('workspace/evaluation-suite-v1-20261003',R/'suite',lambda s:'__pycache__' not in s)
 download('workspace/current-data-v1/run/tokenizer',R/'reference')
 suite=R/'suite';m=json.loads((suite/'manifest.json').read_text())
 for f,h in m['files'].items():assert hashlib.sha256((suite/f).read_bytes()).hexdigest()==h
 save(R/'original-suite-manifest.json',m)
 p=suite/'run_eval.py';s=p.read_text().replace("if not str(HERE).startswith('/data/workspace/'):","if str(HERE)!="+repr(str(suite))+":")
 p.write_text(s);m['files']['run_eval.py']=hashlib.sha256(p.read_bytes()).hexdigest();m['runtime_change']='Authorized training host local working directory; corpus and scoring unchanged';save(suite/'manifest.json',m)
 src=T/'runs'/name/'run';assert json.loads((src/'status.json').read_text())['state']=='trained_calibration_pending'
 cp=json.loads((src/'stage2-selection.json').read_text())['checkpoint'];out=R/'checkpoint';out.mkdir(exist_ok=True)
 for f in ['run.json','stage2-selection.json',cp]:
  dst=out/f
  if not dst.exists():os.link(src/f,dst)
 code=R/'code';code.mkdir(exist_ok=True)
 for f in ['data.py','modeling.py',('adapters_moe_bf16.py' if name=='qwen36-35b-a3b' else 'adapters_short.py')]:shutil.copy2(T/f,code/f)
 (code/'runtime.py').write_text('from pathlib import Path\nROOT=Path('+repr(str(T))+')\nSPACE_CACHE=str(ROOT/"assets")\ndef require_space():\n assert ROOT.is_dir() and str(ROOT) in ["/tmp/pangram-space-fast10","/workspace/woog/pangram/backbones-20261003"]\n')
 # Previously verified inference adapter, same windowing and source-token alignment.
 download('workspace/evaluations-20261003/ettin-code',R/'adapter-template',lambda x:x=='inference.py')
 s=(R/'adapter-template/inference.py').read_text().replace('model.physical_microbatch=8','model.physical_microbatch=1')
 if name=='qwen36-35b-a3b':s=s.replace('from adapters_short import','from adapters_moe_bf16 import')
 (code/'inference.py').write_text(s)
 save(R/'notes.json',{'model':name,'checkpoint':cp,'threshold':0.5,'calibrated':False,'precision':'BF16','reference':'existing ModernBERT tokenizer','training_unchanged':True})
 thread=threading.Thread(target=backup,daemon=True);thread.start()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
 if not h200:env['PYTHONPATH']=str(T/'vendor')
 elif name=='qwen36-35b-a3b':env['PYTHONPATH']=str(T/'moe-vendor')
 else:env.pop('PYTHONPATH',None)
 for phase in ['smoke','full']:
  dest=R/phase;args=[sys.executable,'-u',str(suite/'run_eval.py'),'--run',str(out),'--adapter-dir',str(code),'--reference',str(R/'reference'),'--output',str(dest),'--threshold','0.5','--batch-size','4','--rows-per-chunk','16']
  if phase=='smoke':args+=['--limit','2']
  with (R/(phase+'.log')).open('a') as log:
   p=subprocess.Popen(args,env=env,stdout=log,stderr=log)
   save(R/'worker-status.json',{'phase':phase,'pid':p.pid,'state':'running','time':time.time()})
   rc=p.wait()
  if rc:raise RuntimeError(phase+' failed; return code '+str(rc))
  assert json.loads((dest/'status.json').read_text())['state']=='complete'
 save(R/'worker-status.json',{'state':'complete','time':time.time()});save(R/'finished.json',{'time':time.time()})
 thread.join(timeout=180)
except Exception:
 save(R/'failure.json',{'error':traceback.format_exc()});traceback.print_exc();raise
