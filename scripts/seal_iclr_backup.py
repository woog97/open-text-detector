import hashlib,json,sqlite3,time
import upload_iclr_bulk as u
main=json.loads((u.OUT/'completion.json').read_text());rows={r['path']:r for r in main['files']}
for rel in ['chunks/000007/extraction.json','chunks/000007/runner-state.json']:
 rows[rel]=u.upload((u.SOURCE/rel,rel))
rel='download_queue/queue.sqlite3';snap=u.OUT/'sqlite-snapshots/queue-after-cleanup.sqlite3'
a=sqlite3.connect(f'file:{u.SOURCE/rel}?mode=ro',uri=True);b=sqlite3.connect(snap);a.backup(b);a.close();b.close();rows[rel]=u.upload((snap,rel))
main['files']=list(rows.values());main['previous_manifest']=main.pop('remote_manifest');main['sealed_after_cleanup']=time.time()
raw=json.dumps(main,separators=(',',':')).encode();key='backups/iclr2027/manifests/'+hashlib.sha256(raw).hexdigest()+'.json';receipt=u.put(raw,key)
r=u.client().get(u.URL+key);r.raise_for_status();assert r.content==raw
main['remote_manifest']=receipt;(u.OUT/'completion.json').write_text(json.dumps(main,indent=2))
summary=json.loads((u.OUT/'verification-summary.json').read_text());deletions=[json.loads((u.OUT/name).read_text()) for name in ['local-deletion-summary.json','extraction-deletion-summary.json']]
summary.update(main_manifest=receipt,local_files_deleted=True,deleted_files=sum(d['deleted_files'] for d in deletions),freed_allocated_bytes=sum(d['freed_allocated_bytes'] for d in deletions),changed_source_files_since_snapshot=[],queue_state_backed_up_after_cleanup=True)
summary['total_file_bytes']=sum(r['bytes'] for r in main['files'])+sum(r['bytes'] for r in json.loads((u.OUT/'extraction-completion.json').read_text())['files'])
(out:=u.OUT/'verification-summary.json').write_text(json.dumps(summary,indent=2))
raw=out.read_bytes();key='backups/iclr2027/manifests/'+hashlib.sha256(raw).hexdigest()+'.json';summary_receipt=u.put(raw,key)
(u.OUT/'summary-remote-receipt.json').write_text(json.dumps(summary_receipt,indent=2));print(json.dumps({'deleted_files':summary['deleted_files'],'freed_GiB':summary['freed_allocated_bytes']/1024**3,'main_manifest':receipt['key'],'summary_key':key}))
