"""Attach the shared private W&B dashboard to an already launched Space run.

Run with the existing HF access environment; credentials are passed only in memory.
"""
import argparse
import netrc
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--name', required=True)
args = parser.parse_args()
assert args.root.startswith('/data/workspace/')
sys.path.insert(0, __import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run

key = netrc.netrc().authenticators('api.wandb.ai')[2]
source = Path(__file__).with_name('track.py').read_text()
code = '''
import os,sys,json,subprocess
from pathlib import Path
root=Path(ROOT)
identity_path=root/'wandb-tracker-process.json'
if identity_path.exists():
 d=json.loads(identity_path.read_text());p=Path('/proc',str(d['pid']),'stat')
 if p.exists():
  s=p.read_text().split()
  assert s[2]=='Z' or s[21]!=str(d['start_ticks']), 'Tracker already running'
dest=Path('/data/workspace/wandb-tracking');dest.mkdir(exist_ok=True)
(dest/'track.py').write_text(SOURCE)
if not Path('/tmp/pangram-wandb-vendor/wandb').exists():
 subprocess.run([sys.executable,'-m','pip','install','--target','/tmp/pangram-wandb-vendor','wandb==0.30.0'],check=True,stdout=subprocess.DEVNULL)
env=os.environ.copy()
env.update(WANDB_API_KEY=KEY,WANDB_ENTITY='rigg-alice0',WANDB_PROJECT='open-text-detector-classifiers',WANDB_RUN_GROUP='text-classifiers',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',WANDB_CONSOLE='off')
with (root/'wandb-tracker.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u',str(dest/'track.py'),'--root',str(root),'--name',NAME],cwd='/tmp',env=env,stdout=log,stderr=log,start_new_session=True)
identity={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}
identity_path.write_text(json.dumps(identity))
print(json.dumps(identity))
'''
# No credential-bearing payload is written to disk or printed.
run('ROOT='+repr(args.root)+'\nNAME='+repr(args.name)+'\nSOURCE='+repr(source)+'\nKEY='+repr(key)+'\n'+code)
