import csv,json,statistics,collections,random,datetime
from pathlib import Path
p=Path(__file__).parent
rows=collections.defaultdict(dict)
for r in csv.DictReader(open(p/'minute-gpu-metrics-extended.csv')):rows[r['time_pacific']][int(r['gpu'])]=(float(r['util_pct']),float(r['used_gib']))
obs=[]
for t,gs in sorted(rows.items()):
 dt=datetime.datetime.fromisoformat(t);empty=sum(mem<1 for util,mem in gs.values());quietempty=sum(mem<1 and util<5 for util,mem in gs.values())
 obs.append({'t':t,'day':t[:10],'h':dt.hour,'ts':dt.timestamp(),'util':statistics.mean(v[0] for v in gs.values()),'empty':empty,'quietempty':quietempty,'anyempty':int(empty>0),'twoempty':int(empty>=2),'allempty':int(empty==4)})
def summary(rs):return {k:statistics.mean(r[k] for r in rs) for k in ['util','empty','quietempty','anyempty','twoempty','allempty']}
byday=collections.defaultdict(list)
for r in obs:byday[r['day']].append(r)
days=list(byday);rng=random.Random(4101);boot=collections.defaultdict(list)
for i in range(3000):
 picked=[day for day in rng.choices(days,k=len(days))];n=sum(len(byday[d]) for d in picked)
 for k in ['util','empty','anyempty','twoempty']:
  boot[k].append(sum(sum(r[k] for r in byday[d]) for d in picked)/n)
ci={k:[sorted(v)[int(len(v)*.025)],sorted(v)[int(len(v)*.975)]] for k,v in boot.items()}
# Episodes require adjacent observed minutes. Ends at missing data are censored.
episodes=[]
for gpu in range(4):
 streak=[];last=None
 for t,gs in sorted(rows.items()):
  ts=datetime.datetime.fromisoformat(t).timestamp();qual=gs[gpu][1]<1 and gs[gpu][0]<5
  if streak and (not qual or ts-last>60):episodes.append({'gpu':gpu,'minutes':len(streak),'start':streak[0],'end':streak[-1],'end_observed_busy':not qual and ts-last==60});streak=[]
  if qual:streak.append(t)
  last=ts
 if streak:episodes.append({'gpu':gpu,'minutes':len(streak),'start':streak[0],'end':streak[-1],'end_observed_busy':False})
result={'all_observed':summary(obs),'recent_sep23_24':summary([r for r in obs if r['day']>='2026-09-23']),'day_cluster_bootstrap_95pct_range_conditional_on_observed_days':ci,'four_hour_blocks':[{'hours_pacific':f'{h:02}:00–{h+4:02}:00','minutes':len(rs),'distinct_dates':len(set(r['day'] for r in rs)),**summary(rs)} for h in range(0,24,4) if (rs:=[r for r in obs if h<=r['h']<h+4])],'quiet_nearly_empty_episodes':{'count':len(episodes),'at_least_30min':sum(e['minutes']>=30 for e in episodes),'at_least_60min':sum(e['minutes']>=60 for e in episodes),'longest':sorted(episodes,key=lambda e:e['minutes'],reverse=True)[:8]},'limitations':['Bootstrap reflects variability among 11 observed days, not selection bias or predictive reliability.','A nearly empty card has less than 1 GiB allocated; this is not proof of no processes.','Episodes consist of adjacent sampled minute bins and do not prove inactivity between samples.','Do not infer GPU reservation or permission from estimated spare capacity.']}
json.dump(result,open(p/'capacity-estimates.json','w'),indent=2);print(json.dumps(result,indent=2))
