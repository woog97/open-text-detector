"""Final-run runtime: same interface as backbone-launch runtime.py (ROOT, SPACE_CACHE, require_space)."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parent
HOSTS = ('/tmp/final-20261007', '/workspace/woog/pangram/final-20261007')
sys.path.insert(0, '/tmp/pangram-wandb-vendor')
sys.path.insert(0, str(ROOT / 'vendor'))  # Space: full vendor (transformers 5.19.0); H200: transformers 5.19.0 over the venv
SPACE_CACHE = str(ROOT / 'assets')
def require_space():
    if str(ROOT) not in HOSTS:
        raise RuntimeError('Model operations restricted to authorized training hosts')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
