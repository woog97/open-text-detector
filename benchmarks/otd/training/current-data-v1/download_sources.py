"""Download pinned inputs only to the authorized persistent training Space."""
from pathlib import Path
import os,json,time
from huggingface_hub import snapshot_download
R=Path('/data/workspace/current-data-v1');R.mkdir(exist_ok=True)
def status(**kw):
 p=R/'download-status.json';p.write_text(json.dumps({'time':time.time(),**kw},indent=2))
items=[('human','open-text-detector/human-source-mix-v1','da73a03fdfa863d5c44418e988d50137bcb53c47',['data/*.parquet']),('fullpapers','open-text-detector/synthetic-research-papers-600','65ff939b86908fc8cd8e7bc78d1e5e929c93f1d9',['data/*.parquet']),('papers','woog/ai-paper-provenance-v3','98ca42d9ff340d8a8fddba50e880a8528aafecf2',['data/passages/*.parquet'])]
try:
 for name,repo,revision,patterns in items:
  status(state='downloading',source=name)
  snapshot_download(repo,repo_type='dataset',revision=revision,allow_patterns=patterns,local_dir=R/'sources'/name,max_workers=4)
 status(state='downloading',source='ModernBERT-large')
 snapshot_download('answerdotai/ModernBERT-large',revision='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13',allow_patterns=['*.json','*.safetensors'],local_dir=R/'assets/modernbert',max_workers=4)
 status(state='complete',sources=[x[0] for x in items],model='ModernBERT-large')
except Exception as e:
 status(state='failed',error_type=type(e).__name__,error=str(e));raise
