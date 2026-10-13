"""Select on unchanged validation batches, then calibrate and evaluate once."""
import json,sys,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'comparison';OUT.mkdir(exist_ok=True)
BASE=Path('/data/workspace/paper-batch-sweep-v1/batch32-lr2e4')
SUITE='/data/workspace/paper-v3-modernbert-20260930/eval-suite-v1'
REFERENCE='/data/workspace/paper-v3-modernbert-20260930/run-01/best_model'
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def command(root,label,args):
 with (OUT/(label+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',*args],cwd=root,stdout=f,stderr=subprocess.STDOUT)
  save('status.json',{'state':'running','phase':label,'child_pid':p.pid,'time':time.time()})
  if p.wait():raise RuntimeError(label+' failed')
try:
 plan=json.loads((ROOT/'PROTOCOL.json').read_text());scores=[]
 for trial in plan['trials']:
  r=ROOT/trial['name'];h=json.loads((r/'run/history.json').read_text());s=json.loads((r/'run/status.json').read_text())
  scores.append({'trial':trial['name'],'lr':trial['lr'],'selection_loss':min(x['selection_loss'] for x in h if x['stage']==2),'history':h,'training':s})
 winner=min(scores,key=lambda x:x['selection_loss'])['trial'];control=json.loads((BASE/'run/history.json').read_text())
 threshold=min(x['selection_loss'] for x in control if x['stage']==2)
 for x in scores:
  elapsed=0;reached=None
  for e in x['history']:
   elapsed+=e['epoch_seconds']
   if e['stage']==2 and e['selection_loss']<=threshold and reached is None:reached=elapsed
  x['first_epoch_validation_seconds_to_control_best']=reached
 save('selection.json',{'winner':winner,'criterion':'Lowest unchanged stage2 validation loss; selected before frozen tests','control':str(BASE),'control_history':control,'control_target_loss':threshold,'trials':scores,'caveats':['One seed; microbatch loss weighting and BF16 batch sensitivity change updates.','Historical control timing can differ with system load.','Time-to-quality sampled only at epoch-end validations; null means target not reached.']})
 r=ROOT/winner
 command(r,'calibration',['calibrate.py','--run',str(r/'run'),'--suite',SUITE,'--reference',REFERENCE])
 for profile in ['workflow','comparison']:
  command(r,'evaluation-'+profile,['evaluate.py','--run',str(r/'run'),'--suite',SUITE,'--reference',REFERENCE,'--profile',profile,'--output',str(r/'results'/profile)])
 save('status.json',{'state':'complete','winner':winner,'time':time.time()})
except Exception:
 save('status.json',{'state':'failed','traceback':traceback.format_exc(),'time':time.time()});raise
