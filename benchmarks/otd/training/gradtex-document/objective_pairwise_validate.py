"""Independent read-only publication gate; validates actual data, not status flags."""
from pathlib import Path
import collections,gzip,hashlib,json,sys

def load(path):
 blob=gzip.decompress(Path(path).read_bytes())
 return blob,[json.loads(line) for line in blob.splitlines()]

def validate(root,control,tokenizer):
 root,control=Path(root),Path(control)
 manifest=json.loads((root/'prepared/manifest.json').read_text())
 original=json.loads((control/'prepared/manifest.json').read_text())
 assert set(manifest['files'])==set(original['files']),'Prepared file inventory drift'
 pool_blob,pool=load(root/'paper-pair-pool.jsonl.gz')
 audit=json.loads((root/'paper-pair-audit.json').read_text())
 assert hashlib.sha256(pool_blob).hexdigest()==audit['sha256_uncompressed'],'Audited pair pool drift'
 assert len(pool)==audit['eligible_pairs']
 members={r[role]['id']:r[role] for r in pool for role in ('human','ai')}
 assert len(members)==2*len(pool),'Ambiguous original pair member'
 report={}
 for key,info in manifest['files'].items():
  blob,rows=load(root/'prepared'/(key+'.jsonl.gz'))
  assert hashlib.sha256(blob).hexdigest()==info['sha256'] and len(rows)==info['rows'],key
  if not key.startswith('stage2-'):
   assert info['sha256']==original['files'][key]['sha256'],('Unchanged data drift',key)
   continue
  _,base=load(control/'prepared'/(key+'.jsonl.gz'));assert len(rows)==len(base)==12000
  old_lengths=[len(x)+2 for x in tokenizer([r['text'] for r in base],add_special_tokens=False)['input_ids']]
  lengths=[len(x)+2 for x in tokenizer([r['text'] for r in rows],add_special_tokens=False)['input_ids']]
  seen=set();sources=collections.Counter();paired_tokens=0;pairs=0
  for start in range(0,len(rows),8):
   groups={}
   for i in range(start,min(start+8,len(rows))):
    r=rows[i];pid=r.get('ranking_pair_id')
    if pid is None:
     assert r==base[i],('Unpaired row altered',key,i)
     continue
    assert r.get('verified_pair_relation') is True and r.get('ranking_role') in ('human','ai')
    assert pid not in seen,('Pair ID reused across batches',pid)
    expected=members.get(r['id']);assert expected,('Unknown source pair member',r['id'])
    for field,value in expected.items():
     if field!='ranking_pair_id':assert r.get(field)==value,('Pair source drift',field)
    assert lengths[i]<=512 and abs(lengths[i]-old_lengths[i])<=3
    group=groups.setdefault(pid,{})
    assert r['ranking_role'] not in group,('Duplicate pair role',pid)
    group[r['ranking_role']]=r;paired_tokens+=lengths[i]
   for pid,g in groups.items():
    assert set(g)=={'human','ai'},('Pair crosses batch boundary',pid)
    assert g['human']['paper_id']==g['ai']['paper_id']
    assert g['human']['id'].rsplit('/',1)[0]==g['ai']['id'].rsplit('/',1)[0]
    seen.add(pid);sources[g['human']['paper_id']]+=1;pairs+=1
  share=paired_tokens/sum(lengths)
  assert .099<=share<=.101 and abs(sum(lengths)/sum(old_lengths)-1)<.005
  assert max(sources.values())<=3
  report[key]={'pairs':pairs,'paired_rows':pairs*2,'paired_token_share':share,'processed_tokens':sum(lengths),'control_tokens':sum(old_lengths),'unique_draw_pair_ids':True,'pool_provenance_exact':True,'microbatch_pairs_complete':True,'retained_rows_unchanged':True,'max_pair_draws_per_source':max(sources.values())}
 return report
