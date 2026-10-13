import json,collections,datetime,statistics,csv
from pathlib import Path
from zoneinfo import ZoneInfo
P=Path(__file__).resolve().parent;tz=ZoneInfo('America/Los_Angeles')
def date(ts):return datetime.datetime.fromtimestamp(ts,tz).isoformat()
def pct(a,p):
 a=sorted(a);return a[round((len(a)-1)*p)] if a else None
hosts={r['run']:r.get('host') for r in map(json.loads,open(P/'run-hosts.jsonl'))}
bins=collections.defaultdict(lambda:collections.defaultdict(list));sources=[];times=[]
for line in open(P/'historical-gpu-metrics.jsonl'):
 r=json.loads(line)
 if 'gpu' not in r:sources.append(r);continue
 if hosts.get(r['run'])!='00c559d8728f':continue
 times.append(r['ts'])
 for k,v in r['gpu'].items():
  if k.endswith(('.gpu','.memoryAllocatedBytes')) and isinstance(v,(int,float)):
   bins[(r['ts']//60,k)][r['run']].append(v)
# Equal weight per observed wall-clock minute; average each stream first,
# then median across duplicate streams observing the same device.
minutes=collections.defaultdict(dict)
for (m,k),runs in bins.items():minutes[m][k]=statistics.median(statistics.mean(v) for v in runs.values())
complete={m:v for m,v in minutes.items() if all('gpu.%d.gpu'%g in v and 'gpu.%d.memoryAllocatedBytes'%g in v for g in range(4))}
rows=[]
for g in range(4):
 u=[v['gpu.%d.gpu'%g] for v in complete.values()];mem=[v['gpu.%d.memoryAllocatedBytes'%g]/2**30 for v in complete.values()]
 rows.append({'gpu':g,'mean_util_pct':statistics.mean(u),'median_util_pct':statistics.median(u),'p90_util_pct':pct(u,.9),'median_used_gib':statistics.median(mem),'p90_used_gib':pct(mem,.9),'max_minute_used_gib':max(mem),'under_1gib_pct':100*sum(x<1 for x in mem)/len(mem)})
byday=collections.defaultdict(list)
for m,v in complete.items():byday[date(m*60)[:10]].append(v)
daily=[]
for day,vs in sorted(byday.items()):
 daily.append({'date_pacific':day,'observed_minutes':len(vs),'mean_util_pct':statistics.mean(v['gpu.%d.gpu'%g] for v in vs for g in range(4)),'mean_cards_under_1gib':statistics.mean(sum(v['gpu.%d.memoryAllocatedBytes'%g]<2**30 for g in range(4)) for v in vs)})
capacity={}
for need in [20,40,80]:
 capacity[str(need)]={'fraction_observed_minutes_at_least_one_card':statistics.mean(any(143771/1024-v['gpu.%d.memoryAllocatedBytes'%g]/2**30>=need+10 for g in range(4)) for v in complete.values()),'mean_qualifying_cards':statistics.mean(sum(143771/1024-v['gpu.%d.memoryAllocatedBytes'%g]/2**30>=need+10 for g in range(4)) for v in complete.values())}
inv=json.load(open(P/'shared-log-inventory.json'));recs=inv['records']
logs=[r for r in recs if r['path'].endswith(('.log','.out')) and not any(x in r['path'].lower() for x in ['cftunnel','/http','download','/dl_','wandb/'])]
activity=collections.Counter(r['mtime'][:10] for r in logs)
live=[]
for r in csv.reader(open(P/'live-samples.csv')):
 if not r or r[0].strip()=='timestamp':continue
 live.append(r)
summary={'timezone':'America/Los_Angeles','host_counts':dict(collections.Counter(hosts.values())),'raw_gpu_samples_matched_host':len(times),'tracker_files':len(sources),'tracker_errors':[r for r in sources if r.get('error')],'first_sample':date(min(times)),'last_sample':date(max(times)),'observed_complete_minutes':len(complete),'observed_hours':len(complete)/60,'calendar_span_hours':(max(times)-min(times))/3600,'gpu':rows,'daily':daily,'memory_capacity_scenarios_need_plus_10GiB_headroom':capacity,'all_four_under_1gib_pct':100*statistics.mean(all(v['gpu.%d.memoryAllocatedBytes'%g]<2**30 for g in range(4)) for v in complete.values()),'mean_empty_cards':statistics.mean(sum(v['gpu.%d.memoryAllocatedBytes'%g]<2**30 for g in range(4)) for v in complete.values()),'inventory':{'files_seen':inv['files_seen'],'candidate_records':len(recs),'read_parse_errors':inv['errors'],'earliest_artifact_mtime':min(r['mtime'] for r in recs),'gpu_evidence_logs':sum(r.get('gpu_evidence',False) for r in recs),'oom_logs':sum(r.get('oom',False) for r in recs),'filtered_log_count':len(logs),'filtered_log_modification_days':len(activity),'monthly_filtered_log_modifications':dict(collections.Counter(r['mtime'][:7] for r in logs))},'live':{'rows':len(live),'first':live[0][0] if live else None,'last':live[-1][0] if live else None,'max_gpu_util':max(float(r[2].strip().split()[0]) for r in live),'max_memory_used_mib':max(float(r[4].strip().split()[0]) for r in live)}}
json.dump(summary,open(P/'summary.json','w'),indent=2)
with open(P/'minute-gpu-metrics.csv','w') as f:
 w=csv.writer(f);w.writerow(['time_pacific','gpu','util_pct','used_gib'])
 for m,v in sorted(complete.items()):
  for g in range(4):w.writerow([date(m*60),g,v['gpu.%d.gpu'%g],v['gpu.%d.memoryAllocatedBytes'%g]/2**30])
print(json.dumps(summary,indent=2))
