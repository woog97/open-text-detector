"""Bound stalled checkpoint waits for one identified healthy-source supervisor."""
from pathlib import Path
import fcntl,json,os,signal,time
from datetime import datetime,timezone
root=Path('/tmp/pangram-human-active-20261002/healthy-wikipedia')

def alive(info):
 p=Path('/proc',str(info['pid']),'stat')
 try:
  f=p.read_text().split();return f[21]==info['start_ticks'] and f[2]!='Z'
 except FileNotFoundError:return False

with (root/'checkpoint-watchdog.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 supervisor=json.loads((root/'process.json').read_text());started=(root/'process.json').stat().st_mtime
 while alive(supervisor):
  current=json.loads((root/'process.json').read_text())
  if current['pid']!=supervisor['pid'] or current['start_ticks']!=supervisor['start_ticks']:break
  receipt=root/'checkpoint.json';last=receipt.stat().st_mtime if receipt.exists() else started
  limit=480 if receipt.exists() else 360
  if time.time()-last>limit:
   child=json.loads((root/'worker-process.json').read_text())
   if alive(child):os.kill(child['pid'],signal.SIGTERM)
   time.sleep(3)
   if alive(supervisor):os.kill(supervisor['pid'],signal.SIGTERM)
   status={'state':'checkpoint_deadline_exceeded_own_workers_stopped','supervisor':supervisor,'child':child,'limit_seconds':limit,'updated_at':datetime.now(timezone.utc).isoformat(),'preservation':'Committed database, previous bucket checkpoints and partial backups retained'}
   (root/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True);break
  time.sleep(5)
