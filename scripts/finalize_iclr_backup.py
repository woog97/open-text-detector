import concurrent.futures,hashlib,json,random,time
import upload_iclr_bulk as u
main=json.loads((u.OUT/'completion.json').read_text());extra=json.loads((u.OUT/'extraction-completion.json').read_text())
assert not main['failed'] and not extra['failures'] and not extra['missing_references']
plan=json.loads((u.OUT/'plan.json').read_text());rows={r['path']:r for r in main['files']}
assert len(rows)==len(main['files']) and set(plan['paths'])==set(rows)
allrows=main['files']+extra['files']
for r in allrows:
 assert sum(p['bytes'] for p in r['parts'])==r['bytes']
 assert all(p['verified'] for p in r['parts'])
parts={p['key']:p for r in allrows for p in r['parts']}
rng=random.Random(20261003);sample=rng.sample(list(parts.values()),24)
sample.append(max(parts.values(),key=lambda p:p['bytes']))
def check(p):
 r=u.client().get(u.URL+p['key']);r.raise_for_status();assert len(r.content)==p['bytes'];assert hashlib.sha256(r.content).hexdigest()==p['sha256'];return p['key']
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check,sample))
current={p.relative_to(u.SOURCE).as_posix():p for p in u.SOURCE.rglob('*') if p.is_file() and not p.is_symlink() and not p.name.endswith(('.part','.lock','-wal','-shm')) and '__pycache__' not in p.parts}
new=sorted(set(current)-set(rows));changed=[rel for rel,p in current.items() if rel in rows and p.suffix not in ('.sqlite3','.sqlite','.db') and (p.stat().st_size!=rows[rel]['bytes'] or p.stat().st_mtime_ns!=rows[rel]['mtime_ns'])]
summary={'bucket':'pangram-paper-atlas','main_files':len(main['files']),'associated_extraction_files':len(extra['files']),'total_file_bytes':sum(r['bytes'] for r in allrows),'unique_object_bytes':sum(p['bytes'] for p in parts.values()),'unique_objects':len(parts),'main_manifest':main['remote_manifest'],'extraction_manifest':extra['remote_manifest'],'sample_full_get_sha256_verified':checked,'all_object_sizes_and_checksums_verified':True,'local_files_deleted':False,'new_source_files_since_snapshot':new,'changed_source_files_since_snapshot':changed,'completed_at':time.time()}
(u.OUT/'verification-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('main_manifest','extraction_manifest','sample_full_get_sha256_verified')}),flush=True)
