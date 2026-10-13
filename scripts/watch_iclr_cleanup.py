import sys,time,json
import archive_local_iclr as a
sys.argv.append('--checkpoint')
while not (a.OUT/'completion.json').exists():
 try:a.checkpoint()
 except Exception as e:print('Checkpoint retry:',type(e).__name__,flush=True)
 time.sleep(25)
sys.argv.remove('--checkpoint');a.cleanup()
rows=[json.loads(x) for x in (a.OUT/'deletions.jsonl').read_text().splitlines()];by={r['path']:r for r in rows}
summary={'deleted_files':len(by),'freed_bytes':sum(r['bytes'] for r in by.values()),'freed_allocated_bytes':sum(r['allocated_bytes'] for r in by.values()),'at':time.time()}
(a.OUT/'cleanup-total.json').write_text(json.dumps(summary));print(json.dumps(summary),flush=True)
