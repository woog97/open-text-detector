"""Compare actual outbound traffic while adding useful upload connections."""
import concurrent.futures,json,subprocess,time
import upload_iclr_bulk as u

def sent():
 result={}
 for line in subprocess.check_output(['/usr/sbin/netstat','-ibn'],text=True).splitlines():
  s=line.split()
  if len(s)>=10 and s[0].startswith('en') and '<Link#' in s[2]:result[s[0]]=int(s[-2])
 return result

def rate(before,after,elapsed):return round(sum(max(0,after.get(k,v)-v) for k,v in before.items())/elapsed/1024**2,2)

a=sent();t=time.monotonic();time.sleep(20);b=sent();baseline=rate(a,b,time.monotonic()-t);print(json.dumps({'baseline_outbound_MiB_s':baseline}),flush=True)
plan=json.loads((u.OUT/'plan.json').read_text());done={json.loads(l)['path'] for l in (u.OUT/'receipts.jsonl').read_text().splitlines()}
paths=[rel for rel in plan['paths'] if rel.endswith('.pdf') and rel not in done][-500:]
results=[];failures=[];before=sent();t=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool,(u.OUT/'boost-receipts.jsonl').open('w') as log:
 fs={pool.submit(u.upload,(u.SOURCE/rel,rel)):rel for rel in paths}
 for future in concurrent.futures.as_completed(fs):
  try:r=future.result();results.append(r);log.write(json.dumps(r)+'\n');log.flush()
  except Exception as e:failures.append({'path':fs[future],'error':type(e).__name__})
  if len(results)%100==0:print(json.dumps({'boost_files':len(results),'total':len(paths)}),flush=True)
elapsed=time.monotonic()-t;boost=rate(before,sent(),elapsed)
r={'baseline_connections':40,'boost_connections':56,'baseline_outbound_MiB_s':baseline,'boost_outbound_MiB_s':boost,'boost_seconds':round(elapsed,1),'extra_files':len(results),'failures':failures,'speed_change_percent':round((boost/baseline-1)*100,1)}
(u.OUT/'concurrency-test.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
