"""Explicit public workflow publication, checksummed and verified at commit."""
import json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from suite import HERE,PROJECT
REPO='woog/ai-paper-workflow-eval';PACKAGE=HERE/'hf-package';RECEIPT=HERE/'huggingface_upload.json'
def main():
 token=next(l.split('=',1)[1].strip().strip('"\x27') for l in (PROJECT/'benchmarks/otd/.env.secrets').read_text().splitlines() if l.startswith('HF_TOKEN='))
 api=HfApi(token=token);assert api.whoami()['name']=='woog'
 hashes=json.loads((PACKAGE/'checksums.json').read_text());files=sorted([*hashes,'checksums.json'])
 assert sorted(str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*') if p.is_file())==files
 for name,h in hashes.items():assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==h
 try:existing=api.repo_info(REPO,repo_type='dataset')
 except RepositoryNotFoundError:existing=None
 if existing:
  assert RECEIPT.exists() and json.loads(RECEIPT.read_text())['commit']==existing.sha,'Unexpected remote version'
 else:api.create_repo(REPO,repo_type='dataset',private=False)
 commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(PACKAGE),allow_patterns=files,parent_commit=existing.sha if existing else None,commit_message='Publish frozen paper workflow evaluation and benchmark selection index')
 receipt={'repo_id':REPO,'url':'https://huggingface.co/datasets/'+REPO,'commit':commit.oid,'status':'verification_pending'};RECEIPT.write_text(json.dumps(receipt,indent=2))
 assert not api.repo_info(REPO,repo_type='dataset').private
 for name in files:
  p=hf_hub_download(REPO,name,repo_type='dataset',revision=commit.oid,token=token)
  assert Path(p).read_bytes()==(PACKAGE/name).read_bytes(),name
 from datasets import load_dataset
 receipt['configs']={}
 for config in ['reconstruction','human_controls','assistance','index']:
  ds=load_dataset(REPO,config,revision=commit.oid,token=token)
  receipt['configs'][config]={s:len(d) for s,d in ds.items()}
 receipt['status']='complete';RECEIPT.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
