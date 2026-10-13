import concurrent.futures,json
import upload_iclr_bulk as u
plan=json.loads((u.OUT/'plan.json').read_text());done=set()
for name in ['receipts.jsonl','boost-receipts.jsonl']:
 done.update(json.loads(l)['path'] for l in (u.OUT/name).read_text().splitlines())
paths=[p for p in plan['paths'] if p.endswith('.pdf') and p not in done][-500:]
print('Additional pending PDFs:',len(paths),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool,(u.OUT/'boost2-receipts.jsonl').open('w') as log:
 fs=[pool.submit(u.upload,(u.SOURCE/rel,rel)) for rel in paths]
 for n,f in enumerate(concurrent.futures.as_completed(fs),1):
  r=f.result();log.write(json.dumps(r)+'\n');log.flush()
  if n%100==0 or n==len(paths):print('Extra verified:',n,'/',len(paths),flush=True)
