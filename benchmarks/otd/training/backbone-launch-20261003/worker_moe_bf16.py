from runtime import ROOT
from pathlib import Path
import subprocess,sys,json,os,time,traceback
name=sys.argv[1];r=ROOT/'runs'/name;r.mkdir(parents=True,exist_ok=True)
def save(d):(r/'worker-status.json').write_text(json.dumps({'time':time.time(),**d},indent=2))
try:
 save({'state':'preparing'})
 subprocess.run([sys.executable,'-u',str(ROOT/'prepare_model.py'),name],check=True)
 save({'state':'training'})
 with (r/'training.log').open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',str(ROOT/'train_moe_bf16.py'),name],stdout=f,stderr=f)
  (r/'training-process.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}))
  rc=p.wait()
 if rc:raise RuntimeError('Training failed; see training.log')
 save({'state':'complete','evaluation':'deferred_by_user'})
except Exception as e:save({'state':'failed','error':str(e)});traceback.print_exc();sys.exit(1)
