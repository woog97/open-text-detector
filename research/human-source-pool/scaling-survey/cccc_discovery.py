import sys,pathlib,json,gzip,hashlib,time,collections,sqlite3
old=pathlib.Path('/tmp/pangram-cccc-staging-20261002');base=pathlib.Path('/tmp/pangram-cccc-scaling-20261002')
sys.path.insert(0,str(old/'pipeline'))
import ingest_cccc as c
base.mkdir(exist_ok=True)
mapping,audited,evidence=c.load_evidence(old)
reviewed=set(mapping['canonical_hosts']);mapping['canonical_hosts']=sorted({x.removeprefix('www.') for x in audited})
# Broad routing is DISCOVERY ONLY. It does not admit newly routed domains.
db=sqlite3.connect(base/'discovery.sqlite3');db.executescript('CREATE TABLE IF NOT EXISTS candidates(url TEXT PRIMARY KEY, host TEXT, bin INTEGER, reviewed INTEGER, raw BLOB); CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY,raw BLOB); CREATE TABLE IF NOT EXISTS cursors(file TEXT PRIMARY KEY,rownum INTEGER,done INTEGER);')
counts=collections.Counter(dict(db.execute('SELECT host,count(*) FROM candidates GROUP BY host')));seen={x[0] for x in db.execute('SELECT url FROM candidates')};started=time.monotonic();scanned=0;reasons=collections.Counter()
existing=sqlite3.connect('file:'+str(old/'stage.sqlite3')+'?mode=ro',uri=True);seen.update(x[0] for x in existing.execute('SELECT url FROM cccc_selection'));existing.close()
def status(state,file=None,row=None):
 out={'source':'cccc','state':state,'discovery_only':True,'eligible_admission':False,'scanned_this_run':scanned,'elapsed':round(time.monotonic()-started,1),'file':file,'row':row,'domain_counts':dict(counts),'reasons':dict(reasons),'potential_by_host_bin':[list(r) for r in db.execute('SELECT host,bin,count(*),reviewed FROM candidates GROUP BY host,bin')],'scidev_capture_count':db.execute('SELECT count(*) FROM captures').fetchone()[0],'updated_at':c.now(),'reviewed_hosts':sorted(reviewed),'existing_candidates':5017,'original_shortfall_long':318,'existing_cap_per_site':410,'nominal_10x_target':53350,'reviewed_domain_capacity_ceiling':len(reviewed)*410}
 c.atomic_json(base/'status.json',out)
for archive in sorted((old/'source-downloads/cccc').glob('*.gz')):
 cursor=db.execute('SELECT rownum,done FROM cursors WHERE file=?',(archive.name,)).fetchone()
 if cursor and cursor[1]:continue
 start=cursor[0] if cursor else 0
 manifest=json.loads(archive.with_suffix(archive.suffix+'.manifest.json').read_text())
 assert c.sha_file(archive)==manifest['sha256']
 with gzip.open(archive,'rb') as f:
  for i,line in enumerate(f):
   if i<start:continue
   scanned+=1;r=json.loads(line);meta=r.get('metadata') or {};host=c.host(meta.get('warc_url',''))
   if host in ('scidev.net','www.scidev.net') and str(meta.get('warc_date',''))[:4].isdigit() and int(meta['warc_date'][:4])<2022:
    wrapper={'record':r,'source_dataset':c.REPO,'source_revision':c.REVISION,'source_file':archive.name,'source_row':i,'archive_sha256':manifest['sha256'],'upstream_json_line_sha256':hashlib.sha256(line).hexdigest()}
    db.execute('INSERT OR IGNORE INTO captures VALUES (?,?)',(hashlib.sha256(line).hexdigest(),gzip.compress(json.dumps(wrapper).encode())))
   reason,info=c.route(r,mapping,audited)
   if reason:reasons[reason]+=1
   elif info['canonical_url'] in seen:reasons['already_seen_url']+=1
   elif counts[info['host']]>=410:reasons['discovery_host_bound']+=1
   else:
    reason,pair=c.make_pair(r,line,archive.name,i,manifest['sha256'],mapping,audited,[0,0,0,1])
    if pair:
     record,raw=pair;db.execute('INSERT INTO candidates VALUES (?,?,?,?,?)',(info['canonical_url'],info['host'],record['length_bin'],int(info['host'] in reviewed),gzip.compress(json.dumps({'candidate':record,'original':raw,'discovery_only':True}).encode())));seen.add(info['canonical_url']);counts[info['host']]+=1
    else:reasons[reason]+=1
   if (i+1)%5000==0:
    db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,0)',(archive.name,i+1));db.commit();status('discovering',archive.name,i)
    if time.monotonic()-started>2400:status('bounded_timeout_resumable',archive.name,i);sys.exit(0)
 db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,1)',(archive.name,i+1));db.commit();status('discovering',archive.name,i)
status('complete_discovery_requires_review');db.close()
