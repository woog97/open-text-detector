"""Copy immutable H200 checkpoints to Space, streaming through RAM only.
Run locally with the existing authenticated HF access environment after H200 launch.
No model assets or checkpoints are written to the local computer.
"""
from pathlib import Path
import sys,subprocess,base64,json,time,hashlib,shlex
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from connect import connect
SSH=['ssh','-o','BatchMode=yes','otd-h200']
ROOT='/workspace/woog/pangram/backbones-20261003'
REMOTE='backbone-launch-20261003/h200-checkpoints'
LOCAL=Path(__file__).resolve().parent

def shell(code):return subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)],text=True)
def mkdir(c,p):
 r=c.get('/api/contents/'+p)
 if r.status_code==404:r=c.put('/api/contents/'+p,json={'type':'directory'})
 r.raise_for_status()
def copy(c,name,filename):
 rel=name+'/'+filename;target=REMOTE+'/'+rel
 p=subprocess.Popen(SSH+['cat '+shlex.quote(ROOT+'/runs/'+name+'/run/'+filename)],stdout=subprocess.PIPE)
 h=hashlib.sha256();size=0;chunk=1
 b=p.stdout.read(4*1024*1024)
 while b:
  nxt=p.stdout.read(4*1024*1024);h.update(b);size+=len(b)
  # Always begin with chunk 1; final empty chunk closes a one-piece file as well.
  response=c.put('/api/contents/'+target,json={'type':'file','format':'base64','content':base64.b64encode(b).decode(),'chunk':chunk},timeout=600);response.raise_for_status()
  b=nxt;chunk+=1
 assert p.wait()==0
 response=c.put('/api/contents/'+target,json={'type':'file','format':'base64','content':'','chunk':-1},timeout=600);response.raise_for_status()
 # Full destination readback hash, streamed without storing model files locally.
 check=hashlib.sha256();read_bytes=0
 with c.stream('GET','/files/'+target,timeout=600) as resp:
  resp.raise_for_status()
  for block in resp.iter_bytes(4*1024*1024):check.update(block);read_bytes+=len(block)
 assert check.hexdigest()==h.hexdigest() and read_bytes==size
 return {'file':rel,'bytes':size,'sha256':h.hexdigest(),'space_path':'/data/workspace/'+target,'time':time.time()}

def main():
 import fcntl
 lock=(LOCAL/'checkpoint-preserver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=LOCAL/'h200-checkpoint-preservation.json';saved=json.loads(receipt.read_text()) if receipt.exists() else {};c=None
 while True:
  try:
   if c is None:c=connect();mkdir(c,REMOTE)
   listing=json.loads(shell("from pathlib import Path;import json;r=Path("+repr(ROOT)+");out={};\nfor name in ['gemma4-12b','qwen36-35b-a3b']:\n p=r/'runs'/name/'run';s=p/'status.json';out[name]={'state':json.loads(s.read_text()).get('state') if s.exists() else 'not_started','files':[x.name for x in p.glob('*') if x.is_file() and (x.suffix=='.safetensors' or x.name=='run.json')]};\nprint(json.dumps(out))"))
   for name,d in listing.items():
    if not any(f.endswith('.safetensors') for f in d['files']):continue
    mkdir(c,REMOTE+'/'+name)
    for f in d['files']:
     key=name+'/'+f
     if key not in saved:
      saved[key]=copy(c,name,f);receipt.write_text(json.dumps(saved,indent=2));print(json.dumps(saved[key]),flush=True)
   if all(d['state']=='trained_calibration_pending' for d in listing.values()):return
  except Exception as e:
   print(json.dumps({'time':time.time(),'retrying':type(e).__name__}),flush=True)
   if c is not None:c.close()
   c=None
  time.sleep(60)
if __name__=='__main__':main()
