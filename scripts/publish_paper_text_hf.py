"""Publish a private, versioned Parquet dataset of canonical extracted paper text."""
import hashlib,json,sqlite3,sys
from collections import defaultdict
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
OUT=ROOT/'research/exports/paper-text-hf'
REPO='woog/open-text-detector-paper-text'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def connect(path):
 c=sqlite3.connect(f'file:{ROOT/path}?mode=ro',uri=True);c.row_factory=sqlite3.Row;return c

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
 index=connect('research/extractions/positioned/index.sqlite3')
 maps=[dict(r) for r in index.execute("SELECT * FROM artifacts WHERE kind='canonical' AND mapping_status='complete' AND coverage=1 ORDER BY pdf_sha256")]
 assert len({r['pdf_sha256'] for r in maps})==len(maps)
 catalogue=connect('research/data/reviewbench/catalogue.sqlite3');by_id={};by_forum=defaultdict(list)
 for row in catalogue.execute('SELECT id,title,conference,year FROM papers'):
  d=dict(row);by_id[d['id']]=d;by_forum[d['id'].split(':',1)[-1]].append(d)
 meta={}
 for path in sorted((ROOT/'research/classifications').glob('*/selection.json')):
  for p in json.loads(path.read_text()).get('papers',[]):
   if p.get('pdf_sha256'):meta[p['pdf_sha256']]=p
 queue=connect('research/data/reviewbench_download_queue/queue.sqlite3')
 for p in queue.execute('SELECT id,group_key,sha256 FROM papers WHERE sha256 IS NOT NULL'):
  meta.setdefault(p['sha256'],{}).update(forum_id=p['id'],group=p['group_key'])
 extraction_index=json.loads((ROOT/'research/extractions/new-local-pdfs/extraction-index.json').read_text())
 for p in extraction_index['papers']:
  if p.get('pdfs'):meta.setdefault(p['pdf_sha256'],{}).setdefault('forum_id',Path(p['pdfs'][0]).stem)
 schema=pa.schema([('paper_id',pa.string()),('forum_id',pa.string()),('title',pa.string()),('conference',pa.string()),('year',pa.int32()),('text',pa.large_string()),('pdf_sha256',pa.string()),('text_sha256',pa.string()),('extraction_sha256',pa.string()),('extraction_method',pa.string()),('extraction_metadata_json',pa.string()),('page_count',pa.int32())])
 batch=[];files=[];text_hashes={};missing_metadata=0
 target=OUT/'data'/'batch-001.parquet'
 writer=pq.ParquetWriter(target,schema,compression='zstd',use_dictionary=['conference','year','extraction_method'])
 files.append(target)
 def flush():
  if not batch:return
  for row in batch:assert sha(row['text'].encode())==row['text_sha256']
  writer.write_table(pa.Table.from_pylist(batch,schema=schema),row_group_size=128)
  batch.clear()
 for n,r in enumerate(maps,1):
  raw=(ROOT/r['path']).read_bytes();assert sha(raw)==r['blob_sha256'];d=decode(raw)
  assert d['pdf_sha256']==r['pdf_sha256'] and sha(d['text'].encode())==d['text_sha256']==r['text_sha256']
  m=meta.get(r['pdf_sha256'],{});forum=m.get('forum_id');group=m.get('group','').split('/');conference=group[0] if len(group)==2 else None
  cat=by_id.get(f'{conference}:{forum}')
  if cat is None and len(by_forum.get(forum,[]))==1:cat=by_forum[forum][0]
  cat=cat or {};year=cat.get('year') or (int(group[1]) if len(group)==2 and group[1].isdigit() else None)
  title=cat.get('title') or m.get('title');conference=cat.get('conference') or conference
  if not title:missing_metadata+=1
  quality={k:v for k,v in d.items() if k in ['mapping','diagnostics','warnings','ocr_pages','preprocessing','geometry_version']}
  batch.append(dict(paper_id=r['pdf_sha256'],forum_id=forum,title=title,conference=conference,year=year,text=d['text'],pdf_sha256=r['pdf_sha256'],text_sha256=r['text_sha256'],extraction_sha256=r['blob_sha256'],extraction_method=d.get('method'),extraction_metadata_json=json.dumps(quality,separators=(',',':')),page_count=len(d['pages'])))
  text_hashes[r['pdf_sha256']]=r['text_sha256']
  if len(batch)==1000:flush();print(f'Prepared and verified {n}/{len(maps)} papers',flush=True)
 flush()
 writer.close()
 verified_rows=0
 for chunk in pq.ParquetFile(target).iter_batches(batch_size=128):
  for row in chunk.to_pylist():
   assert sha(row['text'].encode())==text_hashes[row['pdf_sha256']]==row['text_sha256']
   verified_rows+=1
 assert verified_rows==len(maps)
 card='''---
pretty_name: Open Text Detector Paper Text
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
---
# Open Text Detector Paper Text

Canonical extracted text, one row per PDF. PDF/text hashes identify exact versions; metadata records the extraction method and caveats. `train` is the full corpus, not a curated evaluation split. PDFs, position rectangles, and classifier predictions are excluded. Pin a dataset commit for reproducible runs.
'''
 (OUT/'README.md').write_text(card)
 token=next(s.split('=',1)[1].strip().strip('\"\x27') for s in (ROOT/'benchmarks/otd/.env.secrets').read_text().splitlines() if s.startswith('HF_TOKEN='))
 api=HfApi(token=token);assert api.whoami()['name']=='woog'
 try:existing=api.repo_info(REPO,repo_type='dataset')
 except RepositoryNotFoundError:existing=None
 receipt_path=OUT/'upload-receipt.json'
 if existing:
  assert existing.private,'Existing repository is public'
  assert receipt_path.exists() or set(api.list_repo_files(REPO,repo_type='dataset'))<= {'.gitattributes'},'Existing data requires review'
 else:api.create_repo(REPO,repo_type='dataset',private=True)
 names=['README.md']+[p.relative_to(OUT).as_posix() for p in files]
 print(json.dumps({'papers':len(maps),'shards':len(files),'bytes':sum(p.stat().st_size for p in files),'missing_titles':missing_metadata,'repo':REPO}),flush=True)
 commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(OUT),allow_patterns=names,commit_message=f'Add {len(maps):,} canonical paper texts in Parquet shards',parent_commit=existing.sha if existing else None)
 receipt={'repo_id':REPO,'commit':commit.oid,'private':True,'papers':len(maps),'shards':len(files),'verified_files':{}}
 receipt_path.write_text(json.dumps(receipt,indent=2))
 assert api.repo_info(REPO,repo_type='dataset',revision=commit.oid).private
 for n,name in enumerate(names,1):
  remote=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=name,revision=commit.oid,token=token)
  digest=sha((OUT/name).read_bytes());assert sha(Path(remote).read_bytes())==digest
  receipt['verified_files'][name]=digest
  print(f'Verified remote file {n}/{len(names)}',flush=True)
 receipt['verified']=True;receipt_path.write_text(json.dumps(receipt,indent=2))
 print(json.dumps({'url':f'https://huggingface.co/datasets/{REPO}','papers':len(maps),'commit':commit.oid,'private':True,'verified':True}),flush=True)
if __name__=='__main__':main()
