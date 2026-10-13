"""Keep the local MELD worker available until this run drains; persist compact status."""
import collections
import errno
import fcntl
import json
import os
import subprocess
import time
import urllib.error
from process_new_positioned_papers import ROOT,RUN,MANIFEST,api,write,sqlite3,scan_ids,fully_submitted

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    restarts=0;last=None
    try:
        while True:
            try:
                health=api('/health')
                if not health.get('worker_enabled'): raise RuntimeError('Backend worker is disabled')
            except urllib.error.URLError as exc:
                if not isinstance(exc.reason,ConnectionRefusedError):
                    write(RUN/'watchdog-status.json',{'updated_at':time.time(),'state':'waiting_for_backend','error':str(exc)})
                    time.sleep(60);continue
                with (RUN/'backend-recovery.log').open('ab') as log:
                    subprocess.Popen([str(ROOT/'backend/.venv/bin/python'),'-m','uvicorn','open_text_detector_backend.main:app','--host','127.0.0.1','--port','8000'],cwd=ROOT/'backend',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                restarts+=1;time.sleep(30);continue
            manifest=json.loads(MANIFEST.read_text())
            ids={sid for p in manifest['papers'] for sid in scan_ids(p)}
            submitted=sum(fully_submitted(p) for p in manifest['papers'])
            with sqlite3.connect(f'file:{ROOT}/backend/.data/workspace.sqlite3?mode=ro',uri=True) as conn:
                rows=[(r[0],r[1],r[2]) for r in conn.execute('SELECT id,status,error FROM scans') if r[0] in ids]
            counts=dict(collections.Counter(r[1] for r in rows))
            status={'updated_at':time.time(),'population':manifest['population'],'submitted':submitted,'scan_jobs':len(ids),'status':counts,'backend_restarts':restarts,
                'errors':[{'scan_id':r[0],'error':r[2]} for r in rows if r[1]=='failed']}
            write(RUN/'watchdog-status.json',status)
            if counts!=last: print(json.dumps(status),flush=True);last=counts
            if submitted==manifest['population'] and not any(r[1] in ('queued','running') for r in rows):break
            time.sleep(60)
    finally:
        awake.terminate()

if __name__=='__main__':main()
