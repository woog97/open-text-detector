"""Run independent source collectors with bounded concurrency and durable identities.

No model inference, new budgets, or autonomous quota changes. Each source adapter
owns its quota/record validation; one process per source plus one scheduler lock.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from collect_pool import atomic_json, now


def alive(info):
    if not info or not info.get('pid'): return False
    p = Path('/proc', str(info['pid']), 'stat')
    try:
        fields = p.read_text().split()
        return fields[2] != 'Z' and fields[21] == str(info['start_ticks'])
    except FileNotFoundError:
        return False


def source_counts(base):
    db = sqlite3.connect('file:' + str(base / 'collection.sqlite3') + '?mode=ro', uri=True, timeout=30)
    try:return dict(db.execute('SELECT source,count(*) FROM passages GROUP BY source'))
    finally:db.close()


def validate_jobs(jobs, quotas):
    ids = [j['source_id'] for j in jobs]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate source workers are not allowed')
    for j in jobs:
        if j['source_id'] not in quotas: raise ValueError('Source absent from active plan')
        script = j['script']
        if Path(script).name != script or not script.endswith('.py'): raise ValueError('Script must be a pipeline filename')
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=2);p.add_argument('--poll-seconds',type=float,default=5)
    a=p.parse_args()
    if not 1<=a.concurrency<=8: p.error('concurrency must be 1–8')
    b=a.base;folder=b/'parallel';folder.mkdir(exist_ok=True)
    plan=json.loads((b/'pipeline/sampling-plan.json').read_text());quotas={s['source_id']:s['planned_passages'] for s in plan['source_quotas']}
    jobs=validate_jobs(json.loads(a.jobs.read_text())['jobs'],quotas)
    migration=b/'plan-migrations/20261002-drop-yelp-bawe/status.json'
    if not migration.exists() or json.loads(migration.read_text())['state']!='complete':raise RuntimeError('Finish plan migration before source scheduling')
    with (folder/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();baseline=source_counts(b);children={}
        while True:
            counts=source_counts(b);running=[];waiting=[]
            listed={j['source_id'] for j in jobs}
            for path in folder.glob('*-process.json'):
                other=json.loads(path.read_text())
                if other.get('source_id') and other['source_id'] not in listed and alive(other):running.append(other['source_id'])
            for j in jobs:
                sid=j['source_id'];statefile=folder/(sid+'-process.json')
                state=json.loads(statefile.read_text()) if statefile.exists() else {}
                if alive(state):running.append(sid);continue
                if sid in children:
                    rc=children.pop(sid).wait();state.update(returncode=rc,finished_at=now(),elapsed_seconds=round(time.time()-state['started_unix'],3))
                    state['state']='quota_filled' if counts.get(sid,0)>=quotas[sid] else ('source_exhausted' if rc==0 else 'failed')
                    atomic_json(statefile,state)
                elif state.get('state')=='running':
                    state.update(state='needs_reconciliation',observed_dead_at=now());atomic_json(statefile,state)
                if counts.get(sid,0)>=quotas[sid]:continue
                # Failed/exhausted workers need a deliberate source fix, not blind retries.
                if state.get('state') in ('failed','source_exhausted','needs_reconciliation'):continue
                waiting.append(j)
            stop=(folder/'stop-requested.json').exists()
            if not stop:
                for j in waiting[:max(0,a.concurrency-len(running))]:
                    sid=j['source_id'];script=b/'pipeline'/j['script']
                    if not script.is_file():raise RuntimeError('Missing deployed adapter '+str(script))
                    command=[sys.executable,str(script),'--base',str(b),*j.get('args',[])]
                    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    with (folder/(sid+'.log')).open('a') as log:
                        child=subprocess.Popen(command,stdout=log,stderr=log,env=env,start_new_session=True)
                    children[sid]=child
                    state={'source_id':sid,'pid':child.pid,'start_ticks':Path('/proc',str(child.pid),'stat').read_text().split()[21],
                           'state':'running','started_at':now(),'started_unix':time.time(),'command':command,'start_count':counts.get(sid,0),'quota':quotas[sid]}
                    atomic_json(folder/(sid+'-process.json'),state);running.append(sid)
            elapsed=time.monotonic()-started;added=sum(counts.values())-sum(baseline.values())
            unfilled={j['source_id']:quotas[j['source_id']]-counts.get(j['source_id'],0) for j in jobs if counts.get(j['source_id'],0)<quotas[j['source_id']]}
            status={'state':'running' if running else 'stopped' if stop else 'wave_incomplete' if unfilled else 'wave_complete','unfilled_quotas':unfilled,'concurrency':a.concurrency,'running_sources':running,
                    'candidate_total':sum(counts.values()),'wave_added':added,'elapsed_seconds':round(elapsed,2),
                    'observed_pool_records_per_second':round(added/max(elapsed,1),3),'source_counts':counts,'updated_at':now(),
                    'measurement_note':'Pool-wide live progress includes any independently running collector; controlled benchmark is separate.'}
            atomic_json(folder/'status.json',status)
            if not running:print(json.dumps(status),flush=True);break
            time.sleep(a.poll_seconds)

if __name__=='__main__':main()
