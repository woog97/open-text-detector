"""Isolated BF16 throughput trial; original effective batch and validation batching."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
name=sys.argv[1];out=ROOT/name

def save(x):
 p=out/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'time':time.time(),**x},indent=2));t.replace(p)
def command(label,args):
 with (out/(label+'.log')).open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',*args],cwd=out,stdout=f,stderr=subprocess.STDOUT)
  save({'state':'running','phase':label,'child_pid':p.pid})
  if p.wait():raise RuntimeError(label+' failed; preserve partial output and inspect')
try:
 assert not (out/'run').exists(),'No automatic retry or overwrite'
 command('preflight',['preflight.py','--models','causal'])
 command('training',['train.py','--flow','causal','--output',str(out/'run')])
 save({'state':'complete','phase':'trained','training':json.loads((out/'run/status.json').read_text())})
except Exception:
 save({'state':'failed','traceback':traceback.format_exc()});raise
