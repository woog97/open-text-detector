import json, subprocess, sys

checks = {
 'asset_locations': "import os,json;print(json.dumps({p:os.listdir(p) if os.path.isdir(p) else None for p in ['/data/workspace/model-cache','/data/workspace/hf-home/hub','/data/workspace/open-pangram-private-copy','/data/workspace/paper-v3-modernbert-20260930']}))",
 'gradtex_schema': "import pyarrow.parquet as p,json;f=p.ParquetFile('/data/workspace/paper-diversity-v1/public-source-audit/gradtex-train.parquet');print(json.dumps({'rows':f.metadata.num_rows,'columns':f.schema_arrow.names}))",
 'mirrors_schema': "import json; p='/tmp/pangram-luna-active-20261002/stage2-checked-30000-20261002/accepted.jsonl';r=json.loads(open(p).readline());print(json.dumps({'columns':list(r),'types':{k:type(v).__name__ for k,v in r.items()}}))",
 'human_schema': "import sqlite3,json;c=sqlite3.connect('file:/tmp/pangram-human-active-20261002/user-usage-approval-20261003/approved-collection.sqlite3?mode=ro',uri=True);print(json.dumps(c.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()));print(json.dumps(c.execute('PRAGMA table_info(candidates)').fetchall()))",
 'storage_write_probe': "from pathlib import Path;import json;p=Path('/data/workspace/stage4-labels-v1');p.mkdir(exist_ok=True);f=p/'storage-probe.json';f.write_text(json.dumps({'purpose':'stage4 writable-storage probe'}));print(json.dumps({'path':str(f),'readback_ok':json.loads(f.read_text())['purpose']=='stage4 writable-storage probe'}))",
}
for name,code in checks.items():
 try:
  r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=12)
  print(json.dumps({'check':name,'code':r.returncode,'output':r.stdout[:10000],'error':r.stderr[-800:]}),flush=True)
 except subprocess.TimeoutExpired:print(json.dumps({'check':name,'state':'timed_out'}),flush=True)
