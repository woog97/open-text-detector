"""Append approved VOA top-up from the already verified remote original archive."""
from pathlib import Path
import sys,json,sqlite3,gzip,fcntl
B=Path('/tmp/pangram-human-active-20261002');S=Path('/tmp/pangram-voa-staging-20261002');sys.path.insert(0,str(S/'pipeline'))
from ingest_voa import collect
from expand_pool import connect,add
from collect_pool import atomic_json,now
quota=next(x['planned_passages'] for x in json.loads((B/'pipeline/sampling-plan.json').read_text())['source_quotas'] if x['source_id']=='voa');assert quota==1717
collect(S,quota)
s=sqlite3.connect('file:'+str(S/'collection.sqlite3')+'?mode=ro',uri=True);assert s.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
with (B/'voa.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d=connect(B/'collection.sqlite3');before=d.execute('SELECT count(*) FROM passages WHERE source="voa"').fetchone()[0];n=0
 for id,saved in s.execute('SELECT id,row FROM passages WHERE source="voa"'):
  if d.execute('SELECT 1 FROM passages WHERE id=?',(id,)).fetchone():continue
  r=json.loads(saved);raw=json.loads(gzip.decompress(s.execute('SELECT raw FROM documents WHERE hash=?',(r['raw_text_sha256'],)).fetchone()[0]));assert raw['record']['text'][r['raw_start']:r['raw_end']]==r['text']
  with d:n+=add(d,r,raw,quota)
 after=d.execute('SELECT count(*) FROM passages WHERE source="voa"').fetchone()[0];d.close()
s.close();atomic_json(B/'finish100k/voa-merge.json',{'before':before,'added':n,'after':after,'quota':quota,'at':now()})
