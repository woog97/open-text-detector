"""Take the next sweep GPU that frees up for another job, then hand it back.

Holds the sweep queue lock so the first runner to finish a training run blocks
before claiming new work, restricts every unclaimed job to the other GPUs (that
runner then finds nothing and exits), runs the given command on the freed GPU,
and finally removes the restriction and restarts that GPU's runner. Other GPUs
keep training throughout; the deferred job runs on the next GPU to free up.
Usage: claim_sweep_gpu.py --sweeps DIR -- COMMAND... ({gpu} is substituted)
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HOLD='held_for_gpu_job'


def events(log):
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []


def runners():
    found={}
    for proc in Path('/proc').glob('[0-9]*'):
        try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if len(cmd)>2 and cmd[-2].endswith(b'queue_runner.py'):
            found[cmd[-1].decode()]={'pid':int(proc.name),'cmd':[c.decode() for c in cmd if c],'cwd':os.readlink(proc/'cwd')}
        elif len(cmd)>3 and cmd[-3].endswith(b'queue_runner.py'):
            found[cmd[-2].decode()]={'pid':int(proc.name),'cmd':[c.decode() for c in cmd if c],'cwd':os.readlink(proc/'cwd')}
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweeps',required=True,type=Path)
    p.add_argument('--record',required=True,type=Path)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=[c for c in args.command if c!='--']
    S=args.sweeps
    def record(**k):
        previous=json.loads(args.record.read_text()) if args.record.exists() else {}
        args.record.write_text(json.dumps({**previous,**k,'updated_at':time.time()},indent=2))
    before=runners();gpus=sorted(before)
    if not gpus:sys.exit('No sweep runners found')
    # Keep a runner environment (WANDB credentials etc.) for the later restart; never logged.
    runner_env=dict(e.split('=',1) for e in Path(f"/proc/{before[gpus[0]]['pid']}/environ").read_bytes().decode().split('\0') if '=' in e)
    record(state='waiting_for_free_gpu',runners={g:r['pid'] for g,r in before.items()})
    lock=open(S/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
    seen={g:len(events(S/f'runner-gpu{g}.log')) for g in gpus}
    gpu=None
    while gpu is None:
        for g in gpus:
            new=events(S/f'runner-gpu{g}.log')[seen[g]:]
            if any(e.get('event') in ('train_end','stop','queue_empty') for e in new):gpu=g;break
        else:time.sleep(2)
    queue=json.loads((S/'queue.json').read_text())
    (S/f'queue.json.before-gpu{gpu}-hold-{int(time.time())}').write_text(json.dumps(queue,indent=1))
    others=[g for g in gpus if g!=gpu]
    for job in queue:
        if (S/job['tag']/'claim.json').exists():continue
        allowed=job.get('gpus') or gpus
        job[HOLD]={'gpu':gpu,'original_gpus':job.get('gpus')}
        job['gpus']=[g for g in allowed if g!=gpu] or ['none']
    (S/'queue.json').write_text(json.dumps(queue,indent=1))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    record(state='gpu_claimed',gpu=gpu,deferred_jobs=sum(HOLD in j for j in queue))
    # The held runner now finds nothing it may claim and exits.
    end=time.time()+300
    while time.time()<end and gpu in runners():time.sleep(2)
    if gpu in runners():record(state='runner_did_not_exit',gpu=gpu);sys.exit('Runner did not release the GPU')
    record(state='running_job',gpu=gpu,command=[c.replace('{gpu}',gpu) for c in command])
    rc=subprocess.run([c.replace('{gpu}',gpu) for c in command]).returncode
    record(state='restoring_queue',gpu=gpu,job_returncode=rc)
    with open(S/'queue.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=json.loads((S/'queue.json').read_text())
        for job in queue:
            if HOLD in job:
                original=job.pop(HOLD)['original_gpus']
                if original is None:job.pop('gpus',None)
                else:job['gpus']=original
        (S/'queue.json').write_text(json.dumps(queue,indent=1))
    r=before[gpu]
    proc=subprocess.Popen(r['cmd'],cwd=r['cwd'],env=runner_env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    record(state='restored',gpu=gpu,job_returncode=rc,restarted_runner_pid=proc.pid)


if __name__=='__main__':main()
