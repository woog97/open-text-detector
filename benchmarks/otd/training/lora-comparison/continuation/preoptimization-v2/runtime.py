"""Model assets belong exclusively to the persistent Hugging Face Space volume."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'vendor').exists():sys.path.insert(0,str(ROOT/'vendor'))
SPACE_CACHE='/data/workspace/model-cache'
# Set before importing Transformers/Hub in executable modules.
if str(ROOT).startswith('/data/workspace/'):
 os.environ['HF_HUB_CACHE']=SPACE_CACHE
 os.environ['HF_HOME']='/data/workspace/hf-home'
def require_space():
 if not str(ROOT).startswith('/data/workspace/'):
  raise RuntimeError('Model/tokenizer loading is restricted to /data/workspace on the Hugging Face Space. Run this flow there; no local model downloads.')
