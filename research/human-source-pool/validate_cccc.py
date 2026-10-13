"""Validate a frozen CCCC stage, including byte membership in pinned source shards."""
import argparse,collections,gzip,hashlib,json,sqlite3,time
from pathlib import Path
from ingest_cccc import load_evidence,route,sha_file,REVISION,AUDIT_SHA,REPO
from collect_pool import BINS,LENGTH_WEIGHTS,apportion,digest,atomic_json,now


def validate(base,database):
 mapping,audited,evidence=load_evidence(base);d=sqlite3.connect('file:'+str(database)+'?mode=ro&immutable=1',uri=True);assert d.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
 seen_ids=set();seen_text=set();urls=set();hosts=collections.Counter();bins=collections.Counter();captures=collections.Counter();wanted=collections.defaultdict(dict);n=0;begin=time.monotonic()
 for saved, in d.execute("SELECT row FROM passages WHERE source='cccc' ORDER BY id"):
  row=json.loads(saved);raw=json.loads(gzip.decompress(d.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]));original=raw['record'];line=raw['upstream_json_line_utf8'].encode('utf-8');reason,info=route(original,mapping,audited)
  assert reason is None and info['host']==row['publisher_domain']
  assert json.loads(line)==original and hashlib.sha256(line).hexdigest()==raw['upstream_json_line_sha256']
  assert digest(original['text'])==raw['raw_text_sha256']==row['raw_text_sha256']
  assert original['text'][row['raw_start']:row['raw_end']]==row['text'] and digest(row['text'])==row['passage_sha256']
  assert row['source_id']==raw['source_id']=='cccc' and row['category']=='general_web' and not row['training_eligible'] and row['admission_status']=='quarantined_candidate'
  assert row['source_dataset']==raw['source_dataset']==REPO and row['source_revision']==raw['source_revision']==REVISION
  assert row['word_count']==len(row['text'].split());lo,hi=BINS[row['length_bin']];assert lo<=row['word_count']<=hi
  assert row['record_id'] not in seen_ids and info['canonical_url'] not in urls;seen_ids.add(row['record_id']);urls.add(info['canonical_url'])
  normalized=' '.join(row['text'].casefold().split());assert normalized not in seen_text;seen_text.add(normalized)
  assert 'license' not in original['metadata'],'Unexpected nonempty license schema must be reconciled rather than described as absent'
  assert raw['license_evidence']['specific_license_version'] is None and raw['license_evidence']['audit_list_sha256']==AUDIT_SHA
  assert row['license_version_status']=='not_supplied_unverified' and 'exact license/version absent' in row['license_evidence']
  assert row['source_file']==raw['source_file'] and row['source_row']==raw['source_row']
  wanted[row['source_file']][row['source_row']]=(raw['upstream_json_line_sha256'],raw['archive_sha256'])
  hosts[info['host']]+=1;bins[row['length_bin']]+=1;captures[info['capture_date'][:4]]+=1;n+=1
 d.close();assert n<=5335 and max(hosts.values(),default=0)<=410
 targets=apportion(5335,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['general_web'])});assert all(bins[i]<=targets[str(i)] for i in range(4))
 checked=0
 for filename,selected in wanted.items():
  path=base/'source-downloads/cccc'/filename;m=json.loads(path.with_suffix(path.suffix+'.manifest.json').read_text());assert sha_file(path)==m['sha256'];assert m['revision']==REVISION
  assert all(v[1]==m['sha256'] for v in selected.values());found=0;last_wanted=max(selected)
  with gzip.open(path,'rb') as stream:
   for index,line in enumerate(stream):
    if index in selected:assert hashlib.sha256(line).hexdigest()==selected[index][0];found+=1
    if index>=last_wanted:break
  assert found==len(selected);checked+=found;print(json.dumps({'archive_verified':filename,'original_lines_checked':checked,'total':n}),flush=True)
 receipt={'state':'validated_frozen_stage','source_id':'cccc','database':str(database),'database_sha256':sha_file(database),'validated_rows':n,'quota':5335,'domain_counts':dict(hosts),'maximum_per_domain':max(hosts.values(),default=0),'distinct_canonical_post_urls':len(urls),'length_counts':dict(bins),'capture_year_counts':dict(captures),'full_original_archive_sha256_verified':len(wanted),'original_json_line_bytes_membership_verified':checked,'publisher_audit_revision':mapping['github_audit_revision'],'audit_list_sha256':AUDIT_SHA,'specific_document_license_versions':'unknown_not_invented','all_quarantined':True,'global_production_dedup_pending':True,'elapsed_seconds':round(time.monotonic()-begin,2),'updated_at':now()}
 target=base/('validation-'+database.stem+'.json');atomic_json(target,receipt);print('VALIDATED',json.dumps(receipt),flush=True);return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--database',type=Path,required=True);a=p.parse_args();validate(a.base,a.database)
