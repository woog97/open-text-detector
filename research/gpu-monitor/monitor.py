#!/usr/bin/env python3
"""Read-only SSH GPU collector. All persistent files live beside this script."""
import argparse
import collections
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'usage.sqlite3'
TZ = ZoneInfo('America/Los_Angeles')
SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
       '-o', 'ServerAliveCountMax=2', 'otd-h200', 'python3 -u -']

# Passed on stdin; never saved on the server. No command lines, environments,
# private logs, model assets, or research contents are collected.
REMOTE = r'''
import csv,datetime,json,os,pwd,subprocess,time
INTERVAL = __INTERVAL__
ONCE = __ONCE__
def number(s):
 try: return float(s)
 except ValueError: return None
def query(fields,kind):
 p=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 if p.returncode: raise RuntimeError('nvidia-smi '+kind+' failed')
 return list(csv.reader(p.stdout.splitlines(),skipinitialspace=True))
while True:
 started=time.monotonic()
 record={'timestamp':time.time(),'hostname':os.uname().nodename,'gpus':[],'processes':[],'errors':[]}
 try:
  for r in query('index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu','gpu'):
   record['gpus'].append(dict(zip(['index','uuid','util_pct','memory_busy_pct','used_mib','total_mib','power_w','temperature_c'],[int(r[0]),r[1]]+[number(v) for v in r[2:]])))
 except Exception as e:record['errors'].append(type(e).__name__+': gpu query failed')
 try:
  for r in query('gpu_uuid,pid,used_gpu_memory','compute-apps'):
   if not r:continue
   gpu,pid,mem=r;proc={'gpu_uuid':gpu,'pid':int(pid),'used_mib':number(mem),'user':None,'uid':None,'start_ticks':None}
   try:
    uid=os.stat('/proc/'+pid).st_uid;proc['uid']=uid
    try:proc['user']=pwd.getpwuid(uid).pw_name
    except KeyError:proc['user']='uid:'+str(uid)
    stat=open('/proc/'+pid+'/stat').read().rsplit(')',1)[1].split()
    proc['start_ticks']=int(stat[19]);proc['cpu_seconds']=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
   except OSError:proc['owner_status']='unavailable_or_process_exited'
   record['processes'].append(proc)
 except Exception as e:record['errors'].append(type(e).__name__+': process query failed')
 record['collection_seconds']=time.monotonic()-started
 print(json.dumps(record,separators=(',',':')),flush=True)
 if ONCE:break
 time.sleep(max(0,INTERVAL-(time.monotonic()-started)))
'''

def database():
    conn = sqlite3.connect(DB)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, timestamp REAL, payload TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS events (timestamp REAL, kind TEXT, payload TEXT)')
    conn.commit()
    return conn

def overlaps(sample):
    result = {}
    for p in sample['processes']:
        result.setdefault(p['gpu_uuid'], set()).add(p['user'] or 'unknown')
    return {g: sorted(users) for g, users in result.items() if 'woog' in users and any(u != 'woog' for u in users)}

def collect(interval, once):
    if interval < 15:
        raise SystemExit('Use a sampling interval of at least 15 seconds.')
    lock = open(ROOT / 'collector.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A collector is already running.')
    (ROOT / 'collector.pid').write_text(str(os.getpid()))
    conn = database()
    child = None
    def stop(*_):
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    script = REMOTE.replace('__INTERVAL__', str(interval)).replace('__ONCE__', repr(once))
    previous_overlap = {}
    while True:
        child = subprocess.Popen(SSH, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        child.stdin.write(script)
        child.stdin.close()
        received = 0
        for line in child.stdout:
            sample = json.loads(line)
            conn.execute('INSERT INTO samples(timestamp,payload) VALUES (?,?)', (sample['timestamp'], line))
            if not sample['errors']:
                current = overlaps(sample)
                for gpu, users in current.items():
                    if previous_overlap.get(gpu) != users:
                        conn.execute('INSERT INTO events VALUES (?,?,?)', (sample['timestamp'], 'possible_contention_not_proven_failure', json.dumps({'gpu_uuid': gpu, 'users': users})))
                previous_overlap = current
            conn.commit()
            received += 1
            print(dt.datetime.fromtimestamp(sample['timestamp'], TZ).isoformat(), 'sample', len(sample['gpus']), 'GPUs', len(sample['processes']), 'processes', sample['errors'], flush=True)
        code = child.wait()
        if once:
            if code or not received:
                raise SystemExit('No successful SSH collection.')
            return
        conn.execute('INSERT INTO events VALUES (?,?,?)', (time.time(), 'connection_ended', json.dumps({'exit_code': code})))
        conn.commit()
        time.sleep(30)

def summarize(samples):
    daily = collections.defaultdict(lambda: collections.defaultdict(list))
    hourly = collections.defaultdict(lambda: collections.defaultdict(list))
    users = collections.defaultdict(lambda: {'samples_present': 0, 'sum_allocated_gib': 0.0, 'unknown_memory_samples': 0, 'gpu_presence_samples': 0})
    gaps = []
    valid = [s for s in samples if not s['errors']]
    for a,b in zip(samples,samples[1:]):
        if b['timestamp']-a['timestamp']>90:
            gaps.append({'from':a['timestamp'],'to':b['timestamp'],'seconds':b['timestamp']-a['timestamp']})
    for s in valid:
        local = dt.datetime.fromtimestamp(s['timestamp'], TZ)
        for g in s['gpus']:
            daily[local.date().isoformat()][g['index']].append(g)
            hourly[local.strftime('%H:00')][g['index']].append(g)
        byuser = collections.defaultdict(list)
        for p in s['processes']:
            byuser[p['user'] or 'unknown'].append(p)
        for user,ps in byuser.items():
            v = users[user]
            v['samples_present'] += 1
            v['gpu_presence_samples'] += len({p['gpu_uuid'] for p in ps})
            v['sum_allocated_gib'] += sum(p['used_mib'] or 0 for p in ps)/1024
            v['unknown_memory_samples'] += int(any(p['used_mib'] is None for p in ps))
    def aggregate(groups):
        out = {}
        for group, gpus in sorted(groups.items()):
            out[group] = {}
            for gpu,rows in sorted(gpus.items()):
                util = [r['util_pct'] for r in rows if r['util_pct'] is not None]
                mem = [r['used_mib'] for r in rows if r['used_mib'] is not None]
                quiet = [r['used_mib']<1024 and r['util_pct']<5 for r in rows if r['used_mib'] is not None and r['util_pct'] is not None]
                out[group][gpu] = {'samples':len(rows),'mean_util_pct':sum(util)/len(util) if util else None,'mean_used_gib':sum(mem)/len(mem)/1024 if mem else None,'quiet_nearly_empty_fraction':sum(quiet)/len(quiet) if quiet else None}
        return out
    userstats = {u:{'samples_present':v['samples_present'],'fraction_valid_samples_present':v['samples_present']/len(valid),'mean_allocated_gib_while_present':v['sum_allocated_gib']/v['samples_present'] if not v['unknown_memory_samples'] else None,'mean_cards_with_process_while_present':v['gpu_presence_samples']/v['samples_present']} for u,v in users.items()}
    return {'timezone':str(TZ),'total_samples':len(samples),'valid_samples':len(valid),'latest':samples[-1] if samples else None,'daily':aggregate(daily),'hourly':aggregate(hourly),'users':userstats,'gaps_over_90s':gaps,'notes':['Sample-based summaries; missing observations are not idle.','User presence and memory are attributable; whole-GPU compute utilization is not divided among users.','Overlap events indicate possible contention, not proof of a failed or interrupted job.','Unknown process owners remain unknown; no command lines or private job logs are collected.']}

def report():
    conn = database()
    samples = [json.loads(r[0]) for r in conn.execute('SELECT payload FROM samples ORDER BY timestamp,id')]
    result = summarize(samples)
    result['events'] = [{'timestamp':r[0],'kind':r[1],'details':json.loads(r[2])} for r in conn.execute('SELECT * FROM events ORDER BY timestamp')]
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','report'])
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    collect(args.interval,args.once) if args.action=='collect' else report()
