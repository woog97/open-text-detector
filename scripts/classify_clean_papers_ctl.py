"""Start, stop and inspect classify_clean_papers.py workers (one per GPU) on the Space.

  ctl.py JOB.json start 0 1 2     launch a worker on each idle GPU
  ctl.py JOB.json stop 3 4        finish the current shard, then exit (work moves to the others)
  ctl.py JOB.json stop 3 4 --now  kill immediately; the unfinished shard is re-claimed by the others
  ctl.py JOB.json status

JOB.json: {"script", "python_path", "claims", "args": [...], "env": {...}}. Worker
logs and claims stay in the local claims directory; results go to the job's --output.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker_pid(claims,gpu):
    f=claims/'workers'/f'gpu{gpu}.pid'
    if not f.exists():return None
    pid,ticks=f.read_text().split()
    try:fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except OSError:return None
    return int(pid) if fields[0]!='Z' and fields[19]==ticks else None


def gpu_memory(gpu):
    out=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return int(out.stdout)


def main():
    job=json.loads(Path(sys.argv[1]).read_text());command=sys.argv[2];rest=sys.argv[3:]
    claims=Path(job['claims']);(claims/'workers').mkdir(parents=True,exist_ok=True);(claims/'logs').mkdir(exist_ok=True)
    gpus=[int(g) for g in rest if g.isdigit()]
    if command=='start':
        for gpu in gpus:
            if worker_pid(claims,gpu):print(f'gpu{gpu}: already running');continue
            used=gpu_memory(gpu)
            if used>=100:print(f'gpu{gpu}: busy ({used} MiB), skipped');continue
            env=dict(os.environ,**job['env'],CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=job['python_path'])
            env.pop('HF_TOKEN',None)
            with (claims/'logs'/f'gpu{gpu}.log').open('a') as log:
                p=subprocess.Popen([sys.executable,'-u',job['script'],*job['args'],'--claims',str(claims),'--worker',f'gpu{gpu}'],
                                   env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            (claims/'workers'/f'gpu{gpu}.pid').write_text(f'{p.pid} {ticks}')
            print(f'gpu{gpu}: started pid {p.pid}')
    elif command=='stop':
        for gpu in gpus:
            pid=worker_pid(claims,gpu)
            if not pid:print(f'gpu{gpu}: not running');continue
            if '--now' in rest:os.killpg(pid,signal.SIGTERM);print(f'gpu{gpu}: killed pid {pid}')
            else:(claims/f'stop-gpu{gpu}').touch();print(f'gpu{gpu}: will stop after its current shard')
    elif command=='status':
        output=Path(job['args'][job['args'].index('--output')+1])/(job['args'][job['args'].index('--model')+1]+'-fast10')
        if (output/'status.json').exists():print(json.loads((output/'status.json').read_text()))
        for f in sorted((claims/'workers').glob('gpu*.json')):
            gpu=int(f.stem[3:]);s=json.loads(f.read_text())
            print(f.stem,'alive' if worker_pid(claims,gpu) else 'exited',s['state'],
                  {k:s[k] for k in ['shard','shards_done'] if k in s},f"{time.time()-s['updated_at']:.0f}s ago")
    else:sys.exit(__doc__)


if __name__=='__main__':main()
