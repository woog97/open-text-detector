"""One new run: validate BF16, train, attach metrics-only tracking; never resume implicitly."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
R=Path(__file__).resolve().parent

def save(**x):(R/'launch-status.json').write_text(json.dumps({'time':time.time(),**x},indent=2))
def identity(p):return {'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}
def launch(script,args,log,env):
 with (R/log).open('a') as f:return subprocess.Popen([sys.executable,'-u',str(R/script),*args],cwd=R,env=env,stdout=f,stderr=f)
try:
 env=os.environ.copy();model_env={k:v for k,v in env.items() if k!='WANDB_API_KEY'}
 save(state='preflight')
 p=launch('preflight.py',[],'preflight.log',model_env)
 if p.wait():raise RuntimeError('BF16 preflight failed; inspect preflight.log')
 p=launch('train.py',['--flow','encoder','--output',str(R/'run')],'training.log',model_env)
 rec={**identity(p),'started_at':time.time(),'gpu_uuid':env['CUDA_VISIBLE_DEVICES']};(R/'training-process.json').write_text(json.dumps(rec))
 save(state='training_starting',**rec)
 deadline=time.time()+300
 while not (R/'run/run.json').exists():
  if p.poll() is not None:raise RuntimeError('Training exited before run record; inspect training.log')
  if time.time()>deadline:raise RuntimeError('Training still initializing; tracking not attached. Do not restart training; attach tracker separately.')
  time.sleep(2)
 env.update(WANDB_ENTITY='rigg-alice0',WANDB_PROJECT='open-text-detector-classifiers',WANDB_RUN_GROUP='text-classifiers',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
 tracker=launch('track.py',['--root',str(R),'--name',R.name],'wandb-tracker.log',env)
 (R/'wandb-tracker-process.json').write_text(json.dumps(identity(tracker)))
 deadline=time.time()+120
 while not (R/'wandb-tracking.json').exists() and tracker.poll() is None and time.time()<deadline:time.sleep(2)
 tracking=json.loads((R/'wandb-tracking.json').read_text()) if (R/'wandb-tracking.json').exists() else {}
 save(state='training',tracking='attached' if tracking.get('url') else 'not_confirmed',wandb_url=tracking.get('url'),**rec)
 rc=p.wait();save(state='trained' if rc==0 else 'failed',returncode=rc,wandb_url=tracking.get('url'),tracking='attached' if tracking.get('url') else 'not_confirmed')
except Exception as e:save(state='launch_error',error=str(e));traceback.print_exc()
