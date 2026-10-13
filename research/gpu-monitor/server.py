#!/usr/bin/env python3
"""Account-owned, append-only GPU monitor. Requires Python and nvidia-smi only."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from sampler import sample
from summary_core import summarize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PID = ROOT / 'collector.pid'
LOCK = Path('/tmp') / ('gpu-monitor-' + str(os.getuid()) + '.lock')

def append(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, separators=(',', ':')) + '\n')

def running_pid():
    try:
        pid = int(PID.read_text())
        args = Path('/proc') / str(pid) / 'cmdline'
        command = args.read_bytes().split(b'\0')
        return pid if str(ROOT/'server.py').encode() in command and b'run' in command else None
    except (OSError, ValueError):
        return None

def run(interval):
    os.umask(0o077)
    if interval < 15:
        raise SystemExit('Minimum interval is 15 seconds.')
    lock = LOCK.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This account already has a collector running.')
    DATA.mkdir(exist_ok=True)
    PID.write_text(str(os.getpid()))
    active = True
    def stop(*_):
        nonlocal active
        active = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = {}
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    append(DATA/'events.jsonl', {'timestamp':time.time(),'kind':'collector_started','pid':os.getpid(),'interval_seconds':interval})
    while active:
        start = time.monotonic()
        try:
            record = sample()
            record['boot_id'] = boot_id
            record['interval_seconds'] = interval
            date = dt.datetime.fromtimestamp(record['timestamp'], dt.timezone.utc).strftime('%Y-%m-%d')
            append(DATA/('samples-'+date+'.jsonl'), record)
            if not record['errors']:
                current = {}
                for proc in record['processes']:
                    current.setdefault(proc['gpu_uuid'], set()).add(proc['user'] or 'unknown')
                for gpu, users in current.items():
                    if 'woog' in users and len(users)>1 and previous.get(gpu)!=users:
                        append(DATA/'events.jsonl', {'timestamp':record['timestamp'],'kind':'possible_contention_not_proven_failure','gpu_uuid':gpu,'users':sorted(users)})
                previous = current
        except Exception as e:
            # Never touch tenant jobs. Keep trying after collection or storage errors.
            print(dt.datetime.now(dt.timezone.utc).isoformat(),type(e).__name__,str(e),flush=True)
        while active and time.monotonic()-start < interval:
            time.sleep(min(1, max(0, interval-(time.monotonic()-start))))
    append(DATA/'events.jsonl', {'timestamp':time.time(),'kind':'collector_stopped'})

def report():
    samples = []
    invalid = 0
    for path in sorted(DATA.glob('samples-*.jsonl')):
        with path.open() as f:
            for line in f:
                try: samples.append(json.loads(line))
                except ValueError: invalid += 1
    result = summarize(sorted(samples,key=lambda r:r['timestamp']))
    result['ignored_incomplete_lines'] = invalid
    result['collector_pid'] = running_pid()
    result['events'] = []
    if (DATA/'events.jsonl').exists():
        for line in (DATA/'events.jsonl').read_text().splitlines():
            try: result['events'].append(json.loads(line))
            except ValueError: pass
    print(json.dumps(result, indent=2))

def status():
    files = sorted(DATA.glob('samples-*.jsonl'))
    latest = None
    if files:
        with files[-1].open('rb') as f:
            f.seek(max(0,files[-1].stat().st_size-131072))
            for line in f.read().splitlines():
                try: latest = json.loads(line)
                except ValueError: pass
    print(json.dumps({'pid':running_pid(),'latest_sample':latest,'sample_age_seconds':time.time()-latest['timestamp'] if latest else None}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run','report','status','stop','sample'])
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.action=='run': run(args.interval)
    elif args.action=='report': report()
    elif args.action=='status': status()
    elif args.action=='sample': print(json.dumps(sample(),indent=2))
    else:
        pid = running_pid()
        if pid: os.kill(pid,signal.SIGTERM);print('Stop requested for collector',pid)
        else: print('Collector is not running.')
