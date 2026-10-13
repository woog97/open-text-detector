"""Durable runner: extraction gates first, verified R2 cleanup second, MELD queue last."""
import fcntl
import subprocess
import time
from process_new_positioned_papers import ROOT,RUN,write

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in ('extract','finish'):
            write(RUN/'pipeline-status.json',{'updated_at':time.time(),'stage':stage,'state':'running'})
            result=subprocess.run([str(ROOT/'backend/.venv/bin/python'),'-u',str(ROOT/'scripts/process_new_positioned_papers.py'),stage],cwd=ROOT)
            if result.returncode:
                write(RUN/'pipeline-status.json',{'updated_at':time.time(),'stage':stage,'state':'paused_for_review','exit_code':result.returncode})
                return
        write(RUN/'pipeline-status.json',{'updated_at':time.time(),'state':'all_queued'})

if __name__=='__main__':main()
