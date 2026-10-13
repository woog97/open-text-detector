import collections
import datetime as dt
from zoneinfo import ZoneInfo
TZ=ZoneInfo("America/Los_Angeles")

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
