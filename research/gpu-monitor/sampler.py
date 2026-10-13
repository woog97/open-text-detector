
import csv,datetime,json,os,pwd,subprocess,time,glob,collections
_previous_cpu = None
def read_number(path):
 try:return int(open(path).read().strip())
 except (OSError,ValueError):return None
def compact_processes(ranked, users):
 # Poll/rank the top 200, but store only one account rollup and 19 others.
 candidates=ranked[:200]
 others=[p for p in candidates if p['user']!='nev'][:19]
 if len(others)<19:
  others.extend(p for p in ranked[200:] if p['user']!='nev')
  others=others[:19]
 account=users.get('nev',{'process_count':0,'cpu_cores':0.0,'rss_sum_mib':0.0})
 rollup={'user':'nev','kind':'account_total','name':'All visible processes',
         'process_count':account['process_count'],'cpu_cores':account['cpu_cores'],
         'rss_mib':round(account['rss_sum_mib'],2)}
 return [rollup]+others

def cpu_snapshot():
 global _previous_cpu
 now=time.monotonic();hz=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
 stat_lines=open('/proc/stat').readlines()
 ticks=list(map(int,stat_lines[0].split()[1:9]))
 boot_time=int(next(line.split()[1] for line in stat_lines if line.startswith('btime ')))
 uptime=float(open('/proc/uptime').read().split()[0])
 host_total=sum(ticks);host_idle=ticks[3]+ticks[4]
 mem={a[0].rstrip(':'):int(a[1])*1024 for line in open('/proc/meminfo') if len(a:=line.split())>=2 and a[1].isdigit()}
 usage=read_number('/sys/fs/cgroup/cpuacct/cpuacct.usage')
 processes={};users={};unreadable=0
 for path in glob.glob('/proc/[0-9]*'):
  try:
   raw=open(path+'/stat').read();name=raw[raw.index('(')+1:raw.rindex(')')];a=raw.rsplit(')',1)[1].split()
   uid=os.stat(path).st_uid
   try:user=pwd.getpwuid(uid).pw_name
   except KeyError:user='uid:'+str(uid)
   pid=int(path.split('/')[-1]);start=int(a[19]);cpu=(int(a[11])+int(a[12]))/hz;rss=int(a[21])*page/1048576
   key=(pid,start);cores=None
   if _previous_cpu and key in _previous_cpu['processes']:
    cores=max(0,cpu-_previous_cpu['processes'][key])/(now-_previous_cpu['time'])
   r={'pid':pid,'parent_pid':int(a[1]),'name':name,'user':user,'start_ticks':start,'started_at':boot_time+start/hz,'elapsed_seconds':max(0,uptime-start/hz),'cpu_cores':cores,'rss_mib':round(rss,2)}
   processes[key]=(cpu,r)
   u=users.setdefault(user,{'process_count':0,'cpu_cores':0.0 if _previous_cpu else None,'rss_sum_mib':0.0})
   u['process_count']+=1;u['rss_sum_mib']+=rss
   if cores is not None:u['cpu_cores']+=cores
  except (OSError,ValueError,IndexError):unreadable+=1
 host_util=None;container_cores=None
 if _previous_cpu:
  delta=host_total-_previous_cpu['host_total']
  if delta>0:host_util=100*(1-(host_idle-_previous_cpu['host_idle'])/delta)
  if usage is not None and _previous_cpu['usage'] is not None:container_cores=max(0,usage-_previous_cpu['usage'])/1e9/(now-_previous_cpu['time'])
 top=sorted((v[1] for v in processes.values()),key=lambda r:(r['cpu_cores'] or 0,r['rss_mib']),reverse=True)
 result={'host_cpu_util_pct':host_util,'host_logical_cpus':os.cpu_count(),'host_memory_total_gib':mem.get('MemTotal',0)/2**30,'host_memory_available_gib':mem.get('MemAvailable',0)/2**30,'container_cpu_cores':container_cores,'container_memory_bytes':read_number('/sys/fs/cgroup/memory/memory.usage_in_bytes'),'container_memory_limit_bytes':read_number('/sys/fs/cgroup/memory/memory.limit_in_bytes'),'visible_process_count':len(processes),'unreadable_processes':unreadable,'users':users,'top_cpu_processes':compact_processes(top,users),'process_rows_format':'nev_total_plus_19'}
 _previous_cpu={'time':now,'host_total':host_total,'host_idle':host_idle,'usage':usage,'processes':{k:v[0] for k,v in processes.items()}}
 return result
def number(s):
 try: return float(s)
 except ValueError: return None
def query(fields,kind):
 p=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 if p.returncode: raise RuntimeError('nvidia-smi '+kind+' failed')
 return list(csv.reader(p.stdout.splitlines(),skipinitialspace=True))
def sample():
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
 try:record['cpu']=cpu_snapshot()
 except Exception as e:record['cpu_error']=type(e).__name__+': CPU metadata unavailable'
 record['collection_seconds']=time.monotonic()-started
 return record
