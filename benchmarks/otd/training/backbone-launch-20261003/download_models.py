"""Download pinned assets only on the authorized persistent Space storage."""
from pathlib import Path
import json,time,os,concurrent.futures
from huggingface_hub import HfApi,snapshot_download
R=Path('/data/workspace/backbone-launch-20261003');R.mkdir(exist_ok=True)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
MODELS={'ettin-1b':'jhu-clsp/ettin-encoder-1b','qwen35-4b':'Qwen/Qwen3.5-4B','qwen35-9b':'Qwen/Qwen3.5-9B','gemma4-12b':'google/gemma-4-12B','qwen36-27b':'Qwen/Qwen3.6-27B','qwen36-35b-a3b':'Qwen/Qwen3.6-35B-A3B'}
def download(item):
 name,repo=item;p=R/'downloads'/name;p.mkdir(parents=True,exist_ok=True)
 def save(d):(p/'status.json').write_text(json.dumps({'time':time.time(),'repo':repo,**d},indent=2))
 try:
  info=HfApi().model_info(repo);save({'state':'downloading','revision':info.sha})
  dest=R/'assets'/name
  snapshot_download(repo,revision=info.sha,local_dir=dest,allow_patterns=['*.bin','*.safetensors','*.json','*.model','*.txt','*.jinja'],max_workers=4)
  save({'state':'complete','revision':info.sha,'path':str(dest)})
 except Exception as e:save({'state':'failed','error':str(e)[:1800]})
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:list(e.map(download,MODELS.items()))
