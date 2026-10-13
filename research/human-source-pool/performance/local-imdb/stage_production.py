import sys,time,json,hashlib,gzip
from pathlib import Path
HERE=Path(__file__).resolve().parent
PIPE=HERE.parents[1]
sys.path.insert(0,str(PIPE))
from ingest_original_archives import file_hash,collect
from collect_pool import now
from expand_pool import connect
started=time.perf_counter()
base=HERE/'production'
if base.exists():raise RuntimeError('Refuse to overwrite existing production staging')
(base/'pipeline').mkdir(parents=True)
(base/'source-downloads').mkdir()
(base/'source-downloads/imdb').symlink_to(HERE/'source-downloads/imdb',target_is_directory=True)
plan_path=PIPE/'sampling-plan.json';plan=json.loads(plan_path.read_text())
quota=next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id']=='imdb')
assert quota==1061
(base/'pipeline/sampling-plan.json').write_bytes(plan_path.read_bytes())
result=collect(base,'imdb');ingest_wall=time.perf_counter()-started
assert result['count']==quota
package=base/'accepted-pairs.jsonl.gz'
db=connect(base/'collection.sqlite3');seen_ids=set();seen_docs=set();seen_normalized=set();bins={};sentiments={}
with gzip.open(package,'wt',encoding='utf-8',newline='\n') as out:
 for row_text,blob in db.execute('SELECT passages.row,documents.raw FROM passages JOIN documents ON documents.hash=json_extract(passages.row,\'$.raw_text_sha256\') ORDER BY passages.id'):
  row=json.loads(row_text);raw=json.loads(gzip.decompress(blob));text=raw['record']['text'];passage=row['text']
  assert hashlib.sha256(text.encode()).hexdigest()==row['raw_text_sha256']
  assert text[row['raw_start']:row['raw_end']]==passage
  assert hashlib.sha256(passage.encode()).hexdigest()==row['passage_sha256']
  assert row['source_id']=='imdb' and row['category']=='reviews' and row['original_split']=='train'
  assert not row['training_eligible'] and row['admission_status']=='quarantined_candidate'
  norm=' '.join(passage.casefold().split())
  assert row['record_id'] not in seen_ids and row['parent_document_id'] not in seen_docs and norm not in seen_normalized
  seen_ids.add(row['record_id']);seen_docs.add(row['parent_document_id']);seen_normalized.add(norm)
  bins[str(row['length_bin'])]=bins.get(str(row['length_bin']),0)+1
  sentiment=raw['record']['metadata']['sentiment'];sentiments[sentiment]=sentiments.get(sentiment,0)+1
  out.write(json.dumps({'row':row,'raw':raw},ensure_ascii=False,separators=(',',':'))+'\n')
db.close();assert len(seen_ids)==quota
report={'source_id':'imdb','candidate_count':len(seen_ids),'training_admitted':0,'global_merge_pending':True,'quota':quota,
 'plan_sha256':file_hash(plan_path),'archive_manifest':json.loads((HERE/'source-downloads/imdb/manifest.json').read_text()),
 'ingest_status':result,'ingest_wall_seconds':ingest_wall,'preparation_wall_seconds':time.perf_counter()-started,
 'package':str(package),'package_bytes':package.stat().st_size,'package_sha256':file_hash(package),
 'record_ids_sha256':hashlib.sha256('\n'.join(sorted(seen_ids)).encode()).hexdigest(),
 'checks':{'all_original_offsets_and_hashes_verified':True,'unique_document_count':len(seen_docs),'local_normalized_duplicates':0,'source_tags_verified':True},
 'length_bin_counts':bins,'sentiment_counts':sentiments,'created_at':now()}
# Original source download happened during benchmarking; this staging is production candidate material.
report['archive_manifest'].pop('benchmark_only',None)
(base/'package-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
