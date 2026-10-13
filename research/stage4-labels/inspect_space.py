"""Read-only stage-4 inventory. Return metadata only, never dataset text."""
import json, subprocess, sys

checks = {
 'gpu': "import subprocess;print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True));print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True))",
 'workspace': "import os,json;print(json.dumps(os.listdir('/data/workspace')))",
 'models': "import os,json;print(json.dumps({p:os.listdir(p) if os.path.isdir(p) else None for p in ['/data/workspace/models','/data/workspace/hf-cache/hub','/data/huggingface/hub','/root/.cache/huggingface/hub']}))",
 'sources': "import os,json;print(json.dumps({p:os.listdir(p) if os.path.isdir(p) else None for p in ['/data/workspace/paper-diversity-v1/public-source-audit','/data/workspace/paper-backbone-comparison-v1','/tmp/pangram-luna-active-20261002/stage2-checked-30000-20261002']}))",
 'libraries': "import importlib.util,json;print(json.dumps({n:importlib.util.find_spec(n) is not None for n in ['torch','transformers','sentence_transformers','spacy','pyarrow','rapidfuzz']}))",
}
for name, code in checks.items():
 try:
  r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
  print(json.dumps({'check':name,'returncode':r.returncode,'output':r.stdout[:16000],'error':r.stderr[:500]}),flush=True)
 except subprocess.TimeoutExpired:
  print(json.dumps({'check':name,'state':'read_timed_out'}),flush=True)
