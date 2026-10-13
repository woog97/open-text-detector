"""Publish only the requested check annotations to the existing private dataset."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

BASE=Path('/tmp/pangram-human-active-20261002')
AUDIT=BASE/'origin-permission-audit-20261003'
OLD=BASE/'snapshots/human-100000-20261002T194252Z/intake/release'
REPO='open-text-detector/human-source-mix-v1'
PARENT='43c15c7f543690b0a92524c441527f7c4a4129f2'
FIELDS=['human_origin_check_completed','human_origin_check_result','permission_to_use_check_completed','permission_to_use_check_result','stage1_origin_permission_checks_completed','stage1_check_audit_id','stage1_check_details_json']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(token):
    api=HfApi(token=token);info=api.dataset_info(REPO)
    assert info.private and info.sha==PARENT, 'Dataset changed; inspect before publishing'
    summary=json.loads((AUDIT/'summary.json').read_text());assert summary['counts']['checked']==100000
    with gzip.open(AUDIT/'checks.jsonl.gz','rt') as f: checks={r['record_id']:r for r in map(json.loads,f)}
    assert len(checks)==100000
    release=AUDIT/'publication';(release/'data').mkdir(parents=True,exist_ok=True)
    a=release/'audits/origin-permission-20261003';a.mkdir(parents=True,exist_ok=True)
    total=0;ids=set()
    for original in sorted((OLD/'data').glob('*.parquet')):
        table=pq.read_table(original);rowids=table['record_id'].to_pylist()
        assert not ids.intersection(rowids);ids.update(rowids)
        updated=table
        for field in FIELDS:updated=updated.append_column(field,pa.array([checks[i][field] for i in rowids],type=pa.bool_() if field.endswith('_completed') else pa.string()))
        target=release/'data'/original.name;pq.write_table(updated,target,compression='zstd')
        readback=pq.read_table(target)
        assert readback.select(table.column_names).equals(table), 'Prior data changed'
        assert all(readback['human_origin_check_completed'].to_pylist()) and all(readback['permission_to_use_check_completed'].to_pylist())
        total+=len(rowids)
    assert total==100000 and ids==set(checks)
    for name in ['summary.json','checks.jsonl.gz','source-evidence-manifest.json','audit_stage1.py','PROTOCOL.txt']:
        shutil.copy2(AUDIT/name,a/name)
    shutil.make_archive(str(a/'source-policy-evidence'),'gztar',AUDIT,'source-evidence')
    old_card=(OLD/'README.md').read_text()
    note='\n## Completed human-origin and permission checks\n\nAll 100,000 records have both checks attempted. Results are `passed`, `failed`, or `could_not_verify`; task completion is independent of result. English-only scope. See `audits/origin-permission-20261003/summary.json` and the new check columns on every data row. Original text, source fields, historical intake flags and training-admission fields are unchanged. The new check fields supersede historical “audit pending” flags for these two checks only. Other stages were not performed.\n\n'
    # Keep any dataset-card YAML at the beginning.
    if old_card.startswith('---\n'):
        end=old_card.find('\n---',4)+4;card=old_card[:end]+'\n'+note+old_card[end:]
    else:card=note+old_card
    (release/'README.md').write_text(card)
    files=[{'path':str(p.relative_to(release)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(release.rglob('*')) if p.is_file()]
    (a/'publication-files.json').write_text(json.dumps(files,indent=2)+'\n')
    files.append({'path':str((a/'publication-files.json').relative_to(release)),'sha256':sha(a/'publication-files.json'),'bytes':(a/'publication-files.json').stat().st_size})
    manifest=json.loads((OLD/'upload-manifest.json').read_text())
    inventory={f['path']:f for f in manifest['files']}
    inventory.update({f['path']:f for f in files})
    manifest.update(files=list(inventory.values()),origin_permission_checks_completed=100000,
                    audit_id=summary['audit_id'],previous_release_revision=PARENT,
                    validated_at=summary['completed_at'])
    (release/'upload-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append({'path':'upload-manifest.json','sha256':sha(release/'upload-manifest.json'),'bytes':(release/'upload-manifest.json').stat().st_size})
    print('Publishing annotated rows to existing private dataset',flush=True)
    commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(release),parent_commit=PARENT,commit_message='Complete human-origin and permission check attempts for all 100,000 passages')
    revision=commit.oid
    # Persist commit identity before readback: retry must not create duplicate commits.
    receipt={'repo_id':REPO,'revision':revision,'parent_revision':PARENT,'private':True,'rows':total,'state':'committed_readback_pending','files':files}
    (AUDIT/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('COMMITTED',revision,flush=True)
    for i,f in enumerate(files):
        p=hf_hub_download(REPO,f['path'],repo_type='dataset',revision=revision,token=token,local_dir=str(AUDIT/'readback'))
        assert sha(p)==f['sha256'],f['path']
        if (i+1)%10==0:print('Verified published files',i+1,flush=True)
    receipt.update(state='verified',files_verified=len(files),all_prior_data_columns_unchanged=True,all_records_have_both_check_results=True)
    (AUDIT/'publication-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('RECEIPT',json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)

if __name__=='__main__': main(API_TOKEN)
