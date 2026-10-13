import os,json,subprocess,sys,concurrent.futures
from pathlib import Path
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True),flush=True)
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True),flush=True)
print('processes',[(p.name,(p/'comm').read_text().strip()) for p in Path('/proc').iterdir() if p.name.isdigit() and (p/'comm').exists()],flush=True)
checks={
 'mounts':"import os;print(os.listdir('/datasets')); print(os.listdir('/data/workspace/datasets'))",
 'human':"from pathlib import Path;p=Path('/data/workspace/human-source-mix-v2-recovered-20261002');print([str(x) for x in p.glob('*')]);print([str(x) for x in Path('/data/workspace/datasets').glob('*human*')])",
 'models':"from pathlib import Path;p=Path('/data/workspace/model-cache/models--answerdotai--ModernBERT-large/snapshots');print(list(p.glob('*')))",
 'existing':"from pathlib import Path;import json;r=Path('/data/workspace/paper-diversity-v1');print((r/'auto-dispatch/status.json').read_text()[:4000]);print((r/'gradtex-document10-v1/run/status.json').read_text());print([str(p) for p in (r/'gradtex-document10-v1').glob('*.py')])",
 'schemas':"from pathlib import Path;import pyarrow.parquet as pq;paths=list(Path('/datasets/synthetic-mirrors-luna-28120').rglob('*.parquet'))+[Path('/data/workspace/paper-diversity-v1/public-source-audit/gradtex-train.parquet')];print([(str(p),pq.ParquetFile(p).metadata.num_rows,pq.ParquetFile(p).schema.names) for p in paths])"
}
def one(kv):
 k,c=kv
 try:r=subprocess.run([sys.executable,'-c',c],capture_output=True,text=True,timeout=12);return k,r.stdout,r.stderr[:500]
 except subprocess.TimeoutExpired:return k,'timeout'
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e:
 for r in e.map(one,checks.items()):print(json.dumps(r),flush=True)
