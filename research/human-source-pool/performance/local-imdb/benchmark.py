import sys,time,json,hashlib,platform,os,subprocess,gzip
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent
PIPE=HERE.parents[1]
sys.path.insert(0,str(PIPE))
from ingest_original_archives import SOURCES, file_hash, collect
from collect_pool import now
from expand_pool import connect
source=HERE/'source-downloads/imdb';source.mkdir(parents=True,exist_ok=True)
archive=source/'original.tar.gz';spec=SOURCES['imdb'];begin=time.perf_counter()
downloaded=False
if not archive.exists():
 with requests.get(spec['url'],stream=True,timeout=(30,180)) as response:
  response.raise_for_status()
  with archive.with_suffix('.partial').open('wb') as out:
   for data in response.iter_content(4*1024*1024):
    if data:out.write(data)
 archive.with_suffix('.partial').replace(archive);downloaded=True
download_elapsed=time.perf_counter()-begin
begin=time.perf_counter();sha=file_hash(archive);hash_elapsed=time.perf_counter()-begin
assert sha==spec['sha256']
manifest={'source_id':'imdb','url':spec['url'],'sha256':sha,'bytes':archive.stat().st_size,'retrieved_at':now(),'benchmark_only':True}
(source/'manifest.json').write_text(json.dumps(manifest,indent=2))
runs=[]
for run_index in (1,2):
 base=HERE/('run-'+str(run_index))
 if base.exists():raise RuntimeError('Refuse to overwrite existing benchmark '+str(base))
 (base/'pipeline').mkdir(parents=True)
 (base/'source-downloads').mkdir()
 (base/'source-downloads/imdb').symlink_to(source,target_is_directory=True)
 (base/'pipeline/sampling-plan.json').write_text(json.dumps({'source_quotas':[{'source_id':'imdb','planned_passages':250}],'benchmark_only':True}))
 started=time.perf_counter();result=collect(base,'imdb');wall=time.perf_counter()-started
 db=connect(base/'collection.sqlite3');records=[json.loads(r[0]) for r in db.execute('SELECT row FROM passages')]
 assert len(records)==250
 for row in records:
  raw=json.loads(gzip.decompress(db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()[0]))
  assert hashlib.sha256(raw['record']['text'].encode()).hexdigest()==row['raw_text_sha256']
  assert raw['record']['text'][row['raw_start']:row['raw_end']]==row['text']
  assert hashlib.sha256(row['text'].encode()).hexdigest()==row['passage_sha256']
  assert row['source_id']=='imdb' and row['category']=='reviews' and row['original_split']=='train'
  assert not row['training_eligible']
 assert len({r['record_id'] for r in records})==250
 assert len({r['parent_document_id'] for r in records})==250
 ids_sha=hashlib.sha256('\n'.join(sorted(r['record_id'] for r in records)).encode()).hexdigest()
 db.close()
 runs.append({'run_index':run_index,'wall_seconds':wall,'status':result,'offsets_verified':250,'unique_documents':250,'record_ids_sha256':ids_sha,'database_sha256':file_hash(base/'collection.sqlite3'),'benchmark_only':True})
 print(json.dumps({'completed_run':runs[-1]}),flush=True)
host={key:subprocess.run(['sysctl','-n',key],capture_output=True,text=True).stdout.strip() for key in ['machdep.cpu.brand_string','hw.ncpu','hw.memsize']}
report={'source':'imdb','benchmark_only':True,'production_rows_added':0,'host':{'platform':platform.platform(),'machine':platform.machine(),'logical_cpu_count':os.cpu_count(),**host},'downloaded_this_run':downloaded,'download_seconds':download_elapsed,'checksum_verification_seconds':hash_elapsed,'archive_manifest':manifest,'runs':runs,'reported_at':now()}
(HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
