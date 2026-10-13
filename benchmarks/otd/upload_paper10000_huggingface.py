"""Upload only the validated package to a new private Hugging Face dataset."""
import os,json,hashlib
from pathlib import Path
os.environ['HF_HOME']='/tmp/pangram-paper-hf-hub-cache'
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'benchmarks/otd/exports/ai-paper-provenance-v3-10000'
REPO='woog/ai-paper-provenance-v3'
RECEIPT=ROOT/'research/data/paper-gap10000-v3-luna-20260930/huggingface_upload.json'

def main():
 assert json.loads((PACKAGE/'validation.json').read_text())['passed']
 hashes=json.loads((PACKAGE/'file_checksums.json').read_text());files=sorted([*hashes,'file_checksums.json'])
 assert sorted(p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file())==files
 for name,digest in hashes.items():assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==digest,name
 token=next(l.split('=',1)[1].strip().strip('"\x27') for l in (ROOT/'benchmarks/otd/.env.secrets').read_text().splitlines() if l.startswith('HF_TOKEN='))
 api=HfApi(token=token);assert api.whoami()['name']=='woog'
 try:existing=api.repo_info(REPO,repo_type='dataset')
 except RepositoryNotFoundError:existing=None
 if existing:
  assert existing.private,'Refusing to change an existing public repository'
  previous=RECEIPT if RECEIPT.exists() else ROOT/'research/data/paper-gap2500-v3-luna-20260930/huggingface_upload.json'
  prior=json.loads(previous.read_text());assert prior['repo_id']==REPO and existing.sha==prior['commit'],'Remote revision changed; inspect before update'
 else:api.create_repo(REPO,repo_type='dataset',private=True,exist_ok=False)
 commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(PACKAGE),allow_patterns=files,commit_message='Expand v3 corpus to 2000 papers and 10000 paired reconstructions',parent_commit=existing.sha if existing else None)
 receipt={'repo_id':REPO,'url':'https://huggingface.co/datasets/'+REPO,'commit':commit.oid,'private':True,'verified_files':{},'status':'uploaded_verification_pending'}
 RECEIPT.write_text(json.dumps(receipt,indent=2))
 assert api.repo_info(REPO,repo_type='dataset',revision=commit.oid).private
 assert set(api.list_repo_files(REPO,repo_type='dataset',revision=commit.oid))==set(files)|{'.gitattributes'}
 for name in files:
  path=hf_hub_download(REPO,filename=name,repo_type='dataset',revision=commit.oid,token=token,cache_dir='/tmp/pangram-paper-hf-remote-verify')
  data=Path(path).read_bytes();assert data==(PACKAGE/name).read_bytes(),name
  receipt['verified_files'][name]=hashlib.sha256(data).hexdigest()
 from datasets import load_dataset
 receipt['verified_remote_configs']={}
 for config in ['passages','fresh_passages','paragraphs','sentences']:
  ds=load_dataset(REPO,config,revision=commit.oid,token=token,streaming=True)
  assert set(ds)=={'train','validation','test'}
  for split in ds:assert next(iter(ds[split]))['split']==split
  receipt['verified_remote_configs'][config]=list(ds)
 receipt['status']='complete';RECEIPT.write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='verified_files'},indent=2));print('Verified',len(files),'remote files against local package.')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  # Credentials never enter logs, exception messages, or command arguments.
  print(json.dumps({'error_type':type(exc).__name__,'status':getattr(getattr(exc,'response',None),'status_code',None),'message':'Upload or verification failed; retained local package and any receipt.'}));raise SystemExit(1)
