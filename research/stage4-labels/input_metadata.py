import json, sqlite3, subprocess, sys
from pathlib import Path
c=sqlite3.connect('file:/tmp/pangram-human-active-20261002/user-usage-approval-20261003/approved-collection.sqlite3?mode=ro',uri=True)
print('human_columns',json.dumps(c.execute('PRAGMA table_info(passages)').fetchall()),flush=True)
print('human_count',c.execute('SELECT COUNT(*) FROM passages').fetchone()[0],flush=True)
for path in ['/data/workspace/paper-v3-modernbert-20260930/train.py','/data/workspace/paper-backbone-comparison-v1/data.py']:
 try:
  r=subprocess.run([sys.executable,'-c',"from pathlib import Path;import re;print('\\n'.join(x for x in Path("+repr(path)+").read_text().splitlines() if any(k in x for k in ['DATASET=','ROOT=','EXPORT=','cache_dir','snapshot','hf_hub_download','/data/workspace'])))"],capture_output=True,text=True,timeout=8)
  print(json.dumps({'path':path,'output':r.stdout[:4000],'code':r.returncode}),flush=True)
 except subprocess.TimeoutExpired:print('timeout',path,flush=True)
