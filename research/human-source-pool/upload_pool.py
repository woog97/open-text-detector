"""Validate and upload a completed candidate intake to a private HF dataset.

Call upload(base, repo_id, token) from an authenticated session. Never store tokens.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from collect_pool import digest, atomic_json


def validate(base):
 release=base/'intake'/'release'
 summary=json.loads((release/'collection-summary.json').read_text())
 assert summary['admitted_total']==0 and summary['candidate_total']>0
 expected={s['source_id']:s['candidate_passages'] for s in summary['sources']}
 quotas={s['source_id']:s['planned_passages'] for s in summary['sources']}
 registry=json.loads((release/'source-registry.json').read_text())
 category_by_source={s['id']:s['category'] for s in registry['sources']}
 for source in summary['sources']:
  assert source['category']==category_by_source[source['source_id']], 'Summary source/category mismatch'
 seen_ids=set();seen_text=set();counts=Counter()
 for shard in sorted((release/'data').glob('*.parquet')):
  sid=shard.stem;originals={}
  with gzip.open(base/'intake'/'sources'/sid/'documents.jsonl.gz','rt') as stream:
   for line in stream:
    raw=json.loads(line);text=raw['record']['text'];h=digest(text)
    assert h==raw['raw_text_sha256'];originals[h]=text
  for batch in pq.ParquetFile(shard).iter_batches(batch_size=512):
   for row in batch.to_pylist():
    assert row['source_id']==sid and row['admission_status']=='quarantined_candidate' and row['training_eligible'] is False
    assert row.get('category')==category_by_source[sid], 'Missing or mismatched source category'
    assert row['record_id'] not in seen_ids;seen_ids.add(row['record_id'])
    key=' '.join(row['text'].casefold().split());assert key not in seen_text;seen_text.add(key)
    original=originals[row['raw_text_sha256']]
    assert original[row['raw_start']:row['raw_end']]==row['text']
    assert digest(row['text'])==row['passage_sha256']
    assert row['word_count']==len(row['text'].split()) and 50<=row['word_count']<=1500
    assert row['source_revision'] and row['reason_codes'] and row['source_file']
    counts[sid]+=1
  del originals
 for sid,n in expected.items():assert counts[sid]==n and n<=quotas[sid]
 assert sum(counts.values())==summary['candidate_total']
 return summary


def upload(base,repo_id,token):
 base=Path(base);release=base/'intake'/'release';summary=validate(base)
 # Keep originals beside the candidate passages, preserving their attribution.
 (release/'raw').mkdir(exist_ok=True);(release/'pipeline').mkdir(exist_ok=True);(release/'design').mkdir(exist_ok=True)
 for path in sorted((base/'intake'/'sources').glob('*/documents.jsonl.gz')):
  if path.stat().st_size>100:shutil.copy2(path,release/'raw'/(path.parent.name+'.jsonl.gz'))
 for path in (base/'pipeline').glob('*.py'):shutil.copy2(path,release/'pipeline'/path.name)
 for name in ['collection-sources.json','protected-exclusions.json','source-approvals.json','SOURCE_APPROVALS.txt']:
  shutil.copy2(base/'pipeline'/name,release/name)
 for name in ['SAMPLING.md','README.md','SOURCE_REGISTRY.md','evidence-manifest.json']:
  shutil.copy2(base/'pipeline'/name,release/'design'/name)
 # Include source-level evidence and per-adapter immutable download manifests.
 evidence_dir=release/'source-evidence';evidence_dir.mkdir(exist_ok=True)
 for path in sorted((base/'source-downloads').glob('*/manifest.json')):
  shutil.copy2(path,evidence_dir/(path.parent.name+'-manifest.json'))
 for name in ['source-exhaustion.json','sampling-plan-1m.json','rebalance-receipt.json','original-distribution-report.json','quota-reallocation-receipt.json','finish100k-audit.json','ACL_ADAPTER.txt','OANC_SOURCE.txt','OPINRANK_ADAPTER.txt','UBUNTU_IRC_SOURCE.txt']:
  source=base/'pipeline'/name
  if source.exists():shutil.copy2(source,release/'design'/name)
 artifacts=[]
 for path in sorted(release.rglob('*')):
  if path.is_file() and path.name!='upload-manifest.json':
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
   artifacts.append({'path':str(path.relative_to(release)),'sha256':h.hexdigest(),'bytes':path.stat().st_size})
 manifest={'candidate_passages':summary['candidate_total'],'admitted_passages':0,'files':artifacts,'validated_at':datetime.now(timezone.utc).isoformat(),'offsets_and_hashes_verified':True}
 atomic_json(release/'upload-manifest.json',manifest)
 api=HfApi(token=token)
 api.create_repo(repo_id,repo_type='dataset',private=True,exist_ok=True)
 info=api.dataset_info(repo_id)
 if not info.private:raise RuntimeError('Refusing to upload candidates to a public repository')
 commit=api.upload_folder(repo_id=repo_id,repo_type='dataset',folder_path=str(release),commit_message=f"Add {summary['candidate_total']:,} quarantined human-source candidates and provenance")
 revision=commit.oid
 verified=[]
 for artifact in artifacts:
  path=hf_hub_download(repo_id,artifact['path'],repo_type='dataset',revision=revision,token=token,local_dir=str(base/'readback'))
  h=hashlib.sha256()
  with open(path,'rb') as stream:
   for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=artifact['sha256']:raise RuntimeError('Readback checksum mismatch: '+artifact['path'])
  verified.append(artifact['path'])
 receipt={'repo_id':repo_id,'url':f'https://huggingface.co/datasets/{repo_id}','private':True,'revision':revision,'candidate_passages':summary['candidate_total'],'admitted_passages':0,
  'shortfall':summary['shortfall'],'files_verified':len(verified),'bytes_uploaded':sum(f['bytes'] for f in artifacts),'verified_at':datetime.now(timezone.utc).isoformat()}
 atomic_json(base/'upload-receipt.json',receipt);print(json.dumps(receipt),flush=True)
 return receipt
