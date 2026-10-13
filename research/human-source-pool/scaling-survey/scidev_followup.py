"""Bounded followup consuming CCCC's independent historical capture discovery."""
from pathlib import Path
import os,sys,time,json,sqlite3,gzip,shutil,subprocess,hashlib
from datetime import datetime,timezone
old=Path('/tmp/pangram-scidev-scaling-20261002-v1');b=Path('/tmp/pangram-scidev-scaling-20261002-v2');discovery=Path('/tmp/pangram-cccc-scaling-20261002');token=sys.stdin.readline().strip();assert token
end=time.monotonic()+3600
# A tiny queue watcher does no collection while either source worker is running.
def live(pid,ticks):
 p=Path('/proc',str(pid),'stat')
 return p.exists() and p.read_text().split()[21]==ticks and p.read_text().split()[2]!='Z'
while live(193652,'12153674') or live(193753,'12155769'):
 if time.monotonic()>end:raise RuntimeError('Followup wait bound exceeded; inputs preserved')
 time.sleep(15)
b.mkdir(exist_ok=False);shutil.copytree(old/'pipeline',b/'pipeline');shutil.copytree(old/'source-downloads/current-html',b/'source-downloads/current-html');(b/'progress').mkdir()
src=sqlite3.connect('file:'+str(old/'stage.sqlite3')+'?mode=ro',uri=True);out=sqlite3.connect(b/'stage.sqlite3');src.backup(out);src.close();out.close()
d=sqlite3.connect('file:'+str(discovery/'discovery.sqlite3')+'?mode=ro',uri=True);d.execute('BEGIN');seen=set();n=0
with gzip.open(b/'source-downloads/historical-wrappers.jsonl.gz','wb') as dest:
 for row, in d.execute('select raw from captures'):
  raw=gzip.decompress(row);h=hashlib.sha256(raw).hexdigest()
  if h in seen:continue
  seen.add(h);dest.write(raw+b'\n');n+=1
d.close()
(b/'source-downloads/historical-wrappers.manifest.json').write_text(json.dumps({'upstream':str(discovery/'discovery.sqlite3'),'wrapper_rows':n,'sha256':hashlib.sha256((b/'source-downloads/historical-wrappers.jsonl.gz').read_bytes()).hexdigest(),'retrieved_at':datetime.now(timezone.utc).isoformat()}))
code=(b/'pipeline/scidev_worker.py').read_text().replace('scaling/scidev-v1/','scaling/scidev-v2/').replace('scidev-capacity-v1','scidev-capacity-v2');(b/'pipeline/scidev_worker.py').write_text(code)
with (b/'worker.log').open('a') as log:
 p=subprocess.Popen([sys.executable,str(b/'pipeline/scidev_worker.py'),str(b)],stdin=subprocess.PIPE,stdout=log,stderr=log,cwd=b/'pipeline',start_new_session=True)
 p.stdin.write((token+'\n').encode());p.stdin.close()
identity={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'base':str(b),'input_wrappers':n,'stage_only':True,'max_seconds':2700};(b/'process.json').write_text(json.dumps(identity));print(json.dumps(identity),flush=True)
