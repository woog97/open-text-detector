"""Durable download-only supervisor; other workers consume frozen streaming units."""
import fcntl,json,os,shutil,sqlite3,subprocess,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'research/data/openreview_iclr2027_all'
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def held(p):
 with p.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  return False
def main():
 os.chdir(PROJECT)
 lock=(ROOT/'download-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  state={'pid':os.getpid(),'updated_at':time.time()}
  try:
   with sqlite3.connect(f'file:{ROOT}/download_queue/queue.sqlite3?mode=ro',uri=True) as db:
    counts=dict(db.execute('SELECT status,count(*) FROM papers GROUP BY status'))
   state.update(counts=counts,free_bytes=shutil.disk_usage(ROOT).free)
   if held(ROOT/'shared-account.lock'):state['phase']='download_worker_active'
   elif not counts.get('pending',0):state['phase']='no_pending_downloads'
   elif state['free_bytes']<4*1024**3:state['phase']='waiting_for_disk'
   else:
    parents=[]
    for p in (ROOT/'chunks').iterdir():
     if p.is_dir() and p.name.isdigit() and int(p.name)<10000:parents.append(p)
    parents.sort()
    incomplete=[p for p in parents if (p/'timing.json').exists() and not (p/'manifest.json').exists()]
    chunk=incomplete[-1] if incomplete else ROOT/'chunks'/f'{max([int(p.name) for p in parents],default=0)+1:06d}'
    chunk.mkdir(exist_ok=True)
    # Shared-account lock remains authoritative inside the downloader, including cooldown waits.
    write(chunk/'download-only.json',{'enabled':True,'coordinator':str(Path(__file__))})
    if not held(chunk/'extraction-manager.lock'):
     with (chunk/'extraction-manager.log').open('a') as extraction_log:
      extraction=subprocess.Popen([str(PROJECT/'backend/.venv/bin/python'),'-u','scripts/manage_iclr2027_extraction.py','--parent',chunk.name],stdout=extraction_log,stderr=subprocess.STDOUT,start_new_session=True)
     write(chunk/'extraction-manager-launch.json',{'pid':extraction.pid,'at':time.time()})
    state.update(phase='running_download_worker',chunk=chunk.name)
    write(ROOT/'download-monitor-state.json',state)
    with (chunk/'download.log').open('a') as log:
     worker=subprocess.Popen(['/tmp/openreview-benchmark-venv/bin/python','-u','research/tools/openreview_downloader/run_iclr2027_chunk.py','--chunk',chunk.name],stdout=log,stderr=subprocess.STDOUT)
     write(chunk/'download-launch.json',{'pid':worker.pid,'at':time.time()})
     while worker.poll() is None:
      state.update(worker_pid=worker.pid,updated_at=time.time());write(ROOT/'download-monitor-state.json',state);time.sleep(15)
     state.update(returncode=worker.returncode,phase='download_worker_closed')
   write(ROOT/'download-monitor-state.json',state)
  except Exception as e:
   state.update(phase='needs_attention',error=str(e));write(ROOT/'download-monitor-state.json',state)
  time.sleep(30)
if __name__=='__main__':main()
