from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
SPACE_CACHE='/data/workspace/current-data-v1/assets'
def require_space():
 if not str(ROOT).startswith('/data/workspace/'):
  raise RuntimeError('Run model operations on the training Space only')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
