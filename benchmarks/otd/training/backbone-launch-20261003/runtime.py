from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
if str(ROOT).startswith('/data/workspace/'):
 sys.path.insert(0,'/tmp/pangram-wandb-vendor')
 sys.path.insert(0,'/data/workspace/current-data-v1/vendor')
 sys.path.insert(0,str(ROOT/'vendor'))
SPACE_CACHE=str(ROOT/'assets')
def require_space():
 if not str(ROOT).startswith(('/data/workspace/backbone-launch-20261003','/workspace/woog/pangram/backbones-20261003')):
  raise RuntimeError('Model operations restricted to authorized training hosts')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
