"""Document-count ceilings; preserve existing splits and connected families."""
from collections import Counter,defaultdict
from hashlib import sha256
TOTALS={'human':100000,'mirrors':28120,'papers':20000,'gradtex':142214,'fullpapers':600}
STAGE2={'human':.25,'mirrors':.25,'papers':.25,'gradtex':.20,'fullpapers':.05}
def blocked_groups(pools,reserved_ids):
 return {r['group'] for rows in pools.values() for r in rows if str(r['paper_id']) in reserved_ids or str(r['group']) in {'paper:'+x for x in reserved_ids}}
def cap_groups(pools,blocked):
 blocked=set(blocked)
 # Removing whole connected families also removes their linked rows from other pools.
 for source,rows in pools.items():
  groups=defaultdict(int)
  for r in rows:
   if r['group'] not in blocked:groups[r['group']]+=1
  count=sum(groups.values());cap=TOTALS[source]*80//100
  for group in sorted(groups,key=lambda x:sha256(('baseline-mix-v1/'+x).encode()).hexdigest()):
   if count<=cap:break
   blocked.add(group);count-=groups[group]
 return blocked

def validate(pools,schedules,blocked):
 for source,rows in pools.items():
  assert rows and len(rows)<=TOTALS[source]*80//100,(source,len(rows))
  assert not any(r['group'] in blocked for r in rows)
 eligible={s:{r['id'] for r in rows} for s,rows in pools.items()}
 for name,rows in schedules.items():
  mix=STAGE2 if name.startswith('stage2') else {**STAGE2,'papers':.45,'gradtex':0}
  expected={s:round(len(rows)*p) for s,p in mix.items() if p}
  assert dict(Counter(r['dataset'] for r in rows))==expected,name
  assert all(r['id'] in eligible[r['dataset']] and r['group'] not in blocked for r in rows),name
