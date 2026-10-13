"""Model assets belong exclusively to the persistent Hugging Face Space volume."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'vendor').exists():sys.path.insert(0,str(ROOT/'vendor'))
# Optional kernel environment is isolated from completed full-tuning runs.
qwen_config=ROOT/'configs/qwen35.json'
if qwen_config.exists() and json.loads(qwen_config.read_text()).get('fast_linear_attention',False):
 kernel_vendor=ROOT/'continuation/kernel-vendor'
 if not kernel_vendor.exists():raise RuntimeError('Pinned kernel vendor directory missing')
 sys.path.insert(0,str(kernel_vendor))
 os.environ['TRITON_CACHE_DIR']=str(ROOT/'continuation/triton-cache')
 from importlib.metadata import version
 for package,expected in [('fla-core','0.5.2'),('flash-linear-attention','0.5.2'),('einops','0.8.2')]:
  if version(package)!=expected:raise RuntimeError('Kernel dependency drift: '+package)
 from fla.ops.gated_delta_rule import chunk_gated_delta_rule
 if not chunk_gated_delta_rule.__module__.startswith('fla.'):raise RuntimeError('FLA kernel unavailable')
SPACE_CACHE='/data/workspace/model-cache' 
# Set before importing Transformers/Hub in executable modules.
if str(ROOT).startswith('/data/workspace/'):
 os.environ['HF_HUB_CACHE']=SPACE_CACHE
 os.environ['HF_HOME']='/data/workspace/hf-home'
def require_space():
 if not str(ROOT).startswith('/data/workspace/'):
  raise RuntimeError('Model/tokenizer loading is restricted to /data/workspace on the Hugging Face Space. Run this flow there; no local model downloads.')
