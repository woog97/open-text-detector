"""Recover frozen mirror checks, publish private linked dataset, copy paper archive to Space."""
from pathlib import Path, PurePosixPath
import os,json,hashlib,datetime,tarfile,shutil,sqlite3,subprocess,sys,collections,concurrent.futures
from huggingface_hub import HfApi,snapshot_download
import requests
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path('/data/workspace/dataset-publication-20261003')
SCRATCH=Path('/tmp/pangram-mirrors-publication-20261003')
TOKEN=os.environ.get('HF_TOKEN');API=HfApi(token=TOKEN)
HUMAN_REV='da73a03fdfa863d5c44418e988d50137bcb53c47'
MIRROR_REPO='open-text-detector/synthetic-mirrors-luna-28120'
PAPER_REPO='woog/open-text-detector-paper-text';PAPER_REV='a79fcf1ffcb45a20bf6c9728142d0fa974304e95'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def status(kind,state,**kw):save(ROOT/(kind+'-status.json'),dict(state=state,at=stamp(),**kw))
def copy_papers():
 status('papers','downloading',repo=PAPER_REPO,revision=PAPER_REV)
 dest=Path('/data/workspace/datasets/pangram-paper-text');dest.mkdir(parents=True,exist_ok=True)
 snapshot_download(PAPER_REPO,repo_type='dataset',revision=PAPER_REV,token=TOKEN,local_dir=dest,max_workers=4)
 expected=json.loads((ROOT/'paper-upload-receipt.json').read_text())['verified_files'];checks={}
 for name,h in expected.items():
  actual=digest(dest/name);assert actual==h,(name,'hash mismatch');checks[name]=actual
 parquet=list((dest/'data').glob('*.parquet'));rows=sum(pq.ParquetFile(p).metadata.num_rows for p in parquet);assert rows==14561
 receipt=dict(state='verified',repo=PAPER_REPO,revision=PAPER_REV,space_path=str(dest),rows=rows,verified_files=checks,at=stamp(),private_source=True)
 save(ROOT/'papers-receipt.json',receipt);status('papers','verified',rows=rows,path=str(dest));return receipt

def publish_mirrors():
 expected=json.loads((ROOT/'expected-filter-report.json').read_text());stage=SCRATCH/'stage2-checked-30000-20261002';package=Path('/data/workspace/datasets/synthetic-mirrors-luna-28120')
 if not (ROOT/'filter-reproduced-receipt.json').exists():
  assert not stage.exists(),'Existing partial filter output; inspect before retry'
  status('mirrors','recovering_archives')
  arcs=sorted(Path('/data/workspace/synthetic-mirrors-luna-dollar-v1-recovered-20261002/raw-30000-20261002/checkpoints').glob('*.tar.gz'));assert len(arcs)==12
  assert digest(arcs[-1])=='c247191269f54adb3ee6c722675d7eb3c9c2ef2ea2b964b908b7e7fee55de0ff'
  extracted=0
  recovered_complete=len(list((SCRATCH/'run/raw-documents').glob('*.json')))==30000
  for i,arc in ([] if recovered_complete else enumerate(arcs,1)):
   with tarfile.open(arc,'r|gz') as tar:
    for member in tar:
     n=PurePosixPath(member.name)
     if not member.isfile() or n.is_absolute() or '..' in n.parts:continue
     wanted=(len(n.parts)==3 and n.parts[:2] in [('run','calls'),('run','raw-documents')] and n.suffix=='.json') or str(n) in ['run/raw-production-status.json','run/raw-export-receipt.json','run/raw-production-manifest.json']
     if not wanted:continue
     dest=SCRATCH/str(n);dest.parent.mkdir(parents=True,exist_ok=True)
     with tar.extractfile(member) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
     extracted+=1
   status('mirrors','recovering_archives',archives_done=i,archives_total=12,files_replayed=extracted)
  assert len(list((SCRATCH/'run/raw-documents').glob('*.json')))==30000
  dbpath=str(SCRATCH/'verified-parents.sqlite3')
  if not Path(dbpath).exists():
   status('mirrors','downloading_parent_checkpoint_sequentially',expected_bytes=1211031552)
   url='https://huggingface.co/buckets/open-text-detector/training-storage/resolve/workspace/human-source-mix-v2-recovered-20261002/checkpoints/finish100k-active-20261002T193833Z.sqlite3'
   temp=Path(dbpath+'.partial');h=hashlib.sha256();size=0
   with requests.get(url,headers={'Authorization':'Bearer '+TOKEN},stream=True,timeout=(30,180)) as response:
    response.raise_for_status()
    with temp.open('wb') as f:
     for chunk in response.iter_content(4*1024*1024):
      f.write(chunk);h.update(chunk);size+=len(chunk)
      if size%(64*1024*1024)==0:status('mirrors','downloading_parent_checkpoint_sequentially',bytes=size,expected_bytes=1211031552)
   assert h.hexdigest()=='efe8bace75f6e62a9103fc209d3b5123d869583d43d37cde57f1464fffb71f2f'
   assert size==1211031552;temp.replace(dbpath)
  db=sqlite3.connect('file:'+dbpath+'?mode=ro&immutable=1',uri=True)
  n=0
  with (SCRATCH/'raw-production-30000-input.jsonl').open('w') as f:
   for (raw,) in db.execute('select row from passages order by id'):f.write(raw+'\n');n+=1
  db.close();assert n==100000
  status('mirrors','rechecking_frozen_filter',raw_count=30000,source_parent_rows=n)
  env=os.environ.copy();env['MIRROR_RECOVERY_BASE']=str(SCRATCH);env['MIRROR_PARENT_DB']=dbpath
  with (ROOT/'filter.log').open('a') as log:
   proc=subprocess.run([sys.executable,str(ROOT/'filter_completed_batch.py')],cwd=ROOT,env=env,stdout=log,stderr=log)
  assert proc.returncode==0,'Filter failed; inspect persistent filter.log'
  report=json.loads((stage/'report.json').read_text());assert report['accepted']==28120 and report['rejected']==1880
  assert report['rejection_flags_overlapping']==expected['rejection_flags_overlapping'];checks={}
  for name,meta in expected['artifacts'].items():
   actual=digest(stage/name);assert actual==meta['sha256'],(name,'reproduction checksum differs');checks[name]=actual
  assert report['accepted_categories']==expected['accepted_categories']
  save(ROOT/'filter-reproduced-receipt.json',dict(state='byte_identical',accepted=28120,rejected=1880,checksums=checks,parent_input_reconstructed_from=dbpath,raw_archives=len(arcs),at=stamp()))
 # Persist the exact historical accepted file and filter evidence, separate from the publishable schema.
 evidence=ROOT/'recovered-filter';evidence.mkdir(exist_ok=True)
 for name in ['accepted.jsonl','rejected.jsonl','decisions.jsonl','protocol.json','report.json']:
  if not (evidence/name).exists():shutil.copy2(stage/name,evidence/name)
 status('mirrors','packaging',accepted=28120)
 package.mkdir(parents=True,exist_ok=True);(package/'data').mkdir(exist_ok=True)
 rows=[];ids=set();models=collections.Counter();tiers=collections.Counter();categories=collections.Counter();hashes=set()
 for line in (evidence/'accepted.jsonl').open():
  r=json.loads(line);sid=r['source_record_id'];assert sid not in ids;ids.add(sid)
  assert r['filtering_status']=='passed' and r['stage2_automated_qc']['passed'] and not r['stage2_automated_qc']['flags']
  assert r['requested_model']=='openai/gpt-6-luna';text_hash=hashlib.sha256(r['text'].encode()).hexdigest();assert text_hash not in hashes;hashes.add(text_hash)
  row={'record_id':'luna-mirror/'+sid,'source_record_id':sid,'source_dataset':'open-text-detector/human-source-mix-v1','source_dataset_revision':HUMAN_REV,'source_passage_sha256':r['source_passage_sha256'],'source_id':r['source_id'],'category':r['category'],'parent_document_id':r.get('parent_document_id'),'document_family_id':r.get('document_family_id'),'text':r['text'],'text_sha256':text_hash,'authorship':'synthetic','label':1,'model':'openai/gpt-6-luna','requested_model':r['requested_model'],'response_model':r.get('model'),'topic':r.get('topic'),'known_process':r.get('known_process'),'filtering_status':'passed','qc_passed':True,'qc_json':json.dumps(r['stage2_automated_qc'],sort_keys=True),'usage_json':json.dumps(r.get('usage'),sort_keys=True),'requested_service_tier':r.get('requested_service_tier'),'reported_service_tier':r.get('reported_service_tier'),'finish_reason':r.get('finish_reason'),'writer_call_id':r['writer_call_id'],'created_at':r.get('created_at'),'saved_at':r.get('saved_at'),'development_exposed':bool(r.get('development_exposed')),'original_split':r.get('split'),'split_assignment':'unassigned','training_ready':False,'parent_usage_approved':True,'parent_usage_approval_id':'user-all-stage1-usage-20261003-v1','original_admission_status':r.get('admission_status'),'original_training_eligible':r.get('training_eligible')}
  rows.append(row);models[str(row['response_model'])]+=1;tiers[str(row['reported_service_tier'])]+=1;categories[row['category']]+=1
 assert len(rows)==28120
 with (package/'mirrors.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 table=pa.Table.from_pylist(rows);pq.write_table(table,package/'data/train-00000-of-00001.parquet',compression='zstd')
 back=pq.read_table(package/'data/train-00000-of-00001.parquet').to_pylist();assert back==rows
 summary={'rows':len(rows),'authorship':'synthetic','label_mapping':{'0':'human','1':'synthetic'},'response_models':dict(models),'reported_service_tiers':dict(tiers),'categories':dict(categories),'parent_repo':'open-text-detector/human-source-mix-v1','parent_revision':HUMAN_REV,'accepted_raw_export_sha256':expected['artifacts']['accepted.jsonl']['sha256'],'unique_parent_ids':len(ids),'unique_exact_output_hashes':len(hashes),'split_policy':'train is storage only; assign folds by parent family jointly with human corpus','new_model_calls':0,'at':stamp()}
 save(package/'provenance.json',summary);shutil.copy2(ROOT/'filter-reproduced-receipt.json',package/'recovery-verification.json');shutil.copy2(evidence/'protocol.json',package/'filter-protocol.json')
 card='''---
language:
- en
tags:
- synthetic
- ai-text-detection
- provenance
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.parquet
---
# GPT-6-Luna synthetic mirrors

28,120 synthetic English passages retained by the frozen automated checks from 30,000 GPT-6-Luna outputs. Each passage was generated from a short topic and genre/length request derived from a human passage; the writer did not receive that original passage. This is a separate linked collection, not a replacement for the human dataset.

## Join to the human sources

Join `source_record_id` to `record_id` in `open-text-detector/human-source-mix-v1`, pinned to revision `da73a03fdfa863d5c44418e988d50137bcb53c47`. Verify the parent `passage_sha256` against `source_passage_sha256`. Human originals are not duplicated here. All human parent passages have the user's usage approval; original audit findings remain in the parent dataset.

`authorship` is always `synthetic`; `label=1` means synthetic (`0` is reserved for human when combining datasets). `model`/`requested_model` identify `openai/gpt-6-luna`; `response_model` preserves the provider's actual reported model. `reported_service_tier` remains nullable: absent metadata is not proof of Flex execution. Metadata identifiers and labels are not training-text inputs.

## Splits and supervision

The `train` file is a storage convention for the whole release, not a completed train/evaluation partition. `split_assignment=unassigned`, `training_ready=false`, and the original exposure/admission fields are retained. Group each mirror with its human parent and related document family before splitting. The original generation was development-exposed; these rows are not a blind final test set. The document provenance is independently generated AI text; no sentence/clause gold or semantic topic/genre review is asserted.

## Filtering and preservation

Frozen automated checks cover completion, length, visible output-token accounting, copying/repetition, topic validity, parent hash/lineage, saved prompt/response linkage, and normalized exact output deduplication. 28,120 passed and 1,880 were rejected without replacements. The accepted, rejected, decision and protocol exports were reproduced byte-for-byte from persistent raw archives, matching the previously saved SHA256 records. All originals and rejection evidence remain on the training Space; the rejected texts and API request traces are not included in this release.

`mirrors.jsonl` and the Parquet file contain the same records. `provenance.json`, `filter-protocol.json`, `recovery-verification.json`, and `file_checksums.json` record origin and validation. This private dataset card grants no new rights to upstream human sources; original source terms and audit evidence remain attached to the linked human corpus.
'''
 (package/'README.md').write_text(card)
 checks={str(p.relative_to(package)):digest(p) for p in package.rglob('*') if p.is_file() and '.cache' not in p.parts and p.name!='file_checksums.json'};save(package/'file_checksums.json',checks)
 if os.environ.get('PREPARE_ONLY')=='1':
  status('mirrors','prepared_awaiting_authentication',rows=28120,path=str(package));return summary
 status('mirrors','uploading_private_dataset',repo=MIRROR_REPO,rows=28120)
 assert not API.repo_exists(MIRROR_REPO,repo_type='dataset'),'Repository already exists; reconcile before replacing files'
 API.create_repo(MIRROR_REPO,repo_type='dataset',private=True)
 commit=API.upload_folder(repo_id=MIRROR_REPO,repo_type='dataset',folder_path=package,commit_message='Publish 28120 verified GPT-6-Luna mirrors linked to human sources',ignore_patterns=['.cache/**'])
 rev=commit.oid;save(ROOT/'mirrors-upload-commit.json',{'repo':MIRROR_REPO,'revision':rev,'at':stamp()})
 status('mirrors','verifying_remote_readback',revision=rev)
 verify=Path('/tmp/pangram-mirrors-hf-readback-20261003');snapshot_download(MIRROR_REPO,repo_type='dataset',revision=rev,token=TOKEN,local_dir=verify,max_workers=4)
 for name,h in checks.items():assert digest(verify/name)==h,(name,'remote readback mismatch')
 assert digest(verify/'file_checksums.json')==digest(package/'file_checksums.json')
 assert pq.ParquetFile(verify/'data/train-00000-of-00001.parquet').metadata.num_rows==28120
 info=API.dataset_info(MIRROR_REPO,revision=rev);assert info.private
 receipt=dict(state='verified',repo_id=MIRROR_REPO,revision=rev,private=True,rows=28120,space_path=str(package),parent_revision=HUMAN_REV,verified_files={**checks,'file_checksums.json':digest(package/'file_checksums.json')},at=stamp())
 save(ROOT/'mirrors-receipt.json',receipt);status('mirrors','verified',repo=MIRROR_REPO,revision=rev,rows=28120);return receipt

def guarded(name,fn):
 try:return fn()
 except Exception as e:
  status(name,'failed',error_type=type(e).__name__,error=str(e)[:500]);raise
if __name__=='__main__':
 if os.environ.get('PREPARE_ONLY')=='1':
  guarded('mirrors',publish_mirrors);sys.exit(0)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(guarded,'papers',copy_papers),pool.submit(guarded,'mirrors',publish_mirrors)]
  results=[x.result() for x in jobs]
 save(ROOT/'completion.json',{'state':'verified','results':results,'at':stamp()})
