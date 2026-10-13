"""Direct H200 downloads explicitly authorized by user October 3, 2026."""
from pathlib import Path
import json,os,time,concurrent.futures,sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import snapshot_download
R=Path('/workspace/woog/pangram/backbones-20261003')
def download(m):
 p=R/'downloads'/m['name'];p.mkdir(parents=True,exist_ok=True)
 def save(d):
  q=p/'status.tmp';q.write_text(json.dumps({'time':time.time(),'repo':m['repo'],'revision':m['revision'],**d},indent=2));q.replace(p/'status.json')
 try:
  save({'state':'downloading'})
  dest=R/'assets'/m['name']
  snapshot_download(m['repo'],revision=m['revision'],local_dir=dest,allow_patterns=['*.safetensors','*.json','*.model','*.txt','*.jinja'],max_workers=4)
  save({'state':'complete','path':str(dest)})
 except Exception as e:save({'state':'failed','error':str(e)[:1200]})
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:list(e.map(download,[m for m in json.loads((R/'models.json').read_text()) if m['host']=='h200' and m.get('enabled',True) and (len(sys.argv)==1 or m['name'] in sys.argv[1:])]))
