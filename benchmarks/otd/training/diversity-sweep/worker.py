"""One independent GPU per arm; immediately calibrate/evaluate after training."""
import sys,os,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SUITE='/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1'
REFERENCE='/data/workspace/paper-v3-modernbert-20260930/run-01/best_model'
name=sys.argv[1];root=ROOT/name

def save(x):
 p=ROOT/(name+'-status.json');t=p.with_suffix('.tmp');t.write_text(json.dumps({'time':time.time(),**x},indent=2));t.replace(p)
def command(phase,args):
 with (root/(phase+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',*args],cwd=root,stdout=f,stderr=subprocess.STDOUT)
  save({'state':'running','arm':name,'phase':phase,'child_pid':p.pid})
  if p.wait()!=0:raise RuntimeError(phase+' failed; inspect '+str(root/(phase+'.log')))
try:
 while not (root/'prepared/manifest.json').exists():
  rec=json.loads((ROOT/'prepare-process.json').read_text());proc=Path('/proc')/str(rec['pid'])
  if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':raise RuntimeError('Preparation failed')
  save({'state':'waiting_for_preparation','arm':name});time.sleep(15)
 if not (root/'READY.json').exists():command('preflight',['preflight.py','--models','encoder'])
 run=root/'run'
 done=(run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('state')=='trained_calibration_pending'
 if not done:command('training',['train.py','--flow','encoder','--output',str(run)])
 if not (run/'thresholds.json').exists():command('calibration',['calibrate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE])
 for profile in ['workflow','comparison']:
  out=root/'results'/profile
  if not (out/'results.json').exists():command('evaluation-'+profile,['evaluate.py','--run',str(run),'--suite',SUITE,'--reference',REFERENCE,'--profile',profile,'--output',str(out)])
 save({'state':'complete','arm':name})
except Exception as e:save({'state':'failed','arm':name,'error':str(e)});traceback.print_exc();sys.exit(1)
