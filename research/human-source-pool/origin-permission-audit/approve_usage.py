"""Record the user's explicit all-record usage approval; preserve audit findings."""
import datetime
import hashlib
import json
import sqlite3
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

BASE=Path('/tmp/pangram-human-active-20261002')
AUDIT=BASE/'origin-permission-audit-20261003'
OUT=BASE/'user-usage-approval-20261003'
REPO='open-text-detector/human-source-mix-v1'
PARENT='0e444ce91099245f6eaf50ff5dd09cab370e4031'
APPROVAL_ID='user-all-stage1-usage-20261003-v1'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main(token):
    api=HfApi(token=token);info=api.dataset_info(REPO)
    assert info.private and info.sha==PARENT, 'Dataset changed; inspect before writing'
    OUT.mkdir(exist_ok=False)
    release=OUT/'publication';(release/'data').mkdir(parents=True)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    fields={'usage_approved':True,'usage_status':'approved_by_user','usage_approved_by':'user',
            'usage_approved_at':now,'usage_approval_id':APPROVAL_ID}
    approval={'approval_id':APPROVAL_ID,'approved_at':now,'approved_by':'user','rows':100000,
        'scope':'All Stage 1 passages, including passed, failed and could_not_verify audit results.',
        'user_instruction':'I approve the use of all of them, approve all of them for usage',
        'meaning':'Explicit user usage decision. Audit findings and recorded licenses remain separate evidence.',
        'usage_fields':fields,'parent_revision':PARENT}
    checked=AUDIT/'checked-collection.sqlite3'
    expected=json.loads((AUDIT/'summary.json').read_text())['output_sha256'];assert sha(checked)==expected
    source=sqlite3.connect('file:'+str(checked)+'?mode=ro&immutable=1',uri=True)
    target=OUT/'approved-collection.sqlite3';dest=sqlite3.connect(target);source.backup(dest)
    count=0
    for rid,raw in source.execute('SELECT id,row FROM passages'):
        row=json.loads(raw);assert 'usage_approved' not in row;row.update(fields)
        dest.execute('UPDATE passages SET row=? WHERE id=?',(json.dumps(row,ensure_ascii=False),rid));count+=1
    dest.commit();assert count==100000 and dest.execute('PRAGMA integrity_check').fetchone()==('ok',)
    for (rid,raw),(rid2,raw2) in zip(source.execute('SELECT id,row FROM passages ORDER BY id'),dest.execute('SELECT id,row FROM passages ORDER BY id')):
        a=json.loads(raw);b=json.loads(raw2);assert rid==rid2 and all(b[k]==v for k,v in a.items()) and b['usage_approved'] is True
    source.close();dest.close()
    previous=json.loads((AUDIT/'publication-receipt.json').read_text())
    old_hashes={f['path']:f['sha256'] for f in previous['files']}
    seen=set()
    for path in sorted((AUDIT/'publication/data').glob('*.parquet')):
        assert sha(path)==old_hashes['data/'+path.name]
        original=pq.read_table(path);ids=original['record_id'].to_pylist()
        assert not seen.intersection(ids);seen.update(ids);table=original
        for k,v in fields.items():table=table.append_column(k,pa.array([v]*len(table),type=pa.bool_() if isinstance(v,bool) else pa.string()))
        p=release/'data'/path.name;pq.write_table(table,p,compression='zstd');back=pq.read_table(p)
        assert back.select(original.column_names).equals(original) and all(back['usage_approved'].to_pylist())
    assert len(seen)==100000
    approval.update(approved_checkpoint=str(target),approved_checkpoint_sha256=sha(target),all_prior_record_fields_unchanged=True)
    metadata=release/'approvals';metadata.mkdir()
    (metadata/'user-usage-20261003.json').write_text(json.dumps(approval,indent=2)+'\n')
    card=(AUDIT/'publication/README.md').read_text()
    note='\n## User approval for use\n\nThe user explicitly approved all 100,000 Stage 1 passages for use, including every outcome of the origin and permission checks. Every row has `usage_approved=true` and `usage_status=approved_by_user`. See `approvals/user-usage-20261003.json`. Earlier audit findings and intake fields are retained as historical/separate evidence; they do not cancel this explicit user usage decision. This update records approval only and does not run any later training-data stage.\n\n'
    if card.startswith('---\n'):
        end=card.find('\n---',4)+4;card=card[:end]+'\n'+note+card[end:]
    else:card=note+card
    (release/'README.md').write_text(card)
    files=[{'path':str(p.relative_to(release)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(release.rglob('*')) if p.is_file()]
    manifest=json.loads((AUDIT/'publication/upload-manifest.json').read_text());inventory={f['path']:f for f in manifest['files']}
    inventory.update({f['path']:f for f in files});manifest.update(files=list(inventory.values()),user_usage_approved_records=100000,usage_approval_id=APPROVAL_ID,previous_release_revision=PARENT,validated_at=now)
    (release/'upload-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append({'path':'upload-manifest.json','sha256':sha(release/'upload-manifest.json'),'bytes':(release/'upload-manifest.json').stat().st_size})
    print('All 100000 rows marked user-approved; publishing',flush=True)
    commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=str(release),parent_commit=PARENT,commit_message='Record explicit user approval for use of all 100,000 Stage 1 passages')
    receipt={**approval,'repo_id':REPO,'revision':commit.oid,'private':True,'state':'committed_readback_pending','files':files}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('COMMITTED',commit.oid,flush=True)
    for i,f in enumerate(files):
        p=hf_hub_download(REPO,f['path'],repo_type='dataset',revision=commit.oid,token=token,local_dir=str(OUT/'readback'))
        assert sha(p)==f['sha256']
        if (i+1)%10==0:print('Verified files',i+1,flush=True)
    receipt.update(state='verified',files_verified=len(files))
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('RECEIPT',json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)

if __name__=='__main__':main(API_TOKEN)
