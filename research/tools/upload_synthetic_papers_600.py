"""Upload the user's 600 final manuscripts to the existing organization."""
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HOME'] = '/tmp/pangram-synthetic600-hf-cache'
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'research/data/synthetic-papers-600-hf-20261003'
REPO = 'open-text-detector/synthetic-research-papers-600'
RECEIPT = PACKAGE.parent / 'synthetic-papers-600-huggingface-upload.json'

def main():
    checksums = json.loads((PACKAGE / 'file_checksums.json').read_text())
    assert json.loads((PACKAGE / 'validation.json').read_text())['checks_passed']
    for name, sha in checksums.items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == sha
    # User explicitly authorized this existing local login after project token
    # lacked organization write permission. Never print or persist its value.
    token = (Path.home() / '.cache/huggingface/token').read_text().strip()
    api = HfApi(token=token)
    account = api.whoami()
    orgs = {org['name']: org for org in account.get('orgs', [])}
    assert 'open-text-detector' in orgs, 'Organization membership not confirmed'
    print(json.dumps({'account': account['name'], 'organization': 'open-text-detector', 'repo_id': REPO}), flush=True)
    try:
        info = api.repo_info(REPO, repo_type='dataset')
    except RepositoryNotFoundError:
        info = None
    if info:
        assert info.private
        files = api.list_repo_files(REPO, repo_type='dataset')
        assert files == ['.gitattributes'] or (RECEIPT.exists() and json.loads(RECEIPT.read_text())['commit'] == info.sha), 'Existing dataset requires inspection before replacement'
    else:
        api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=False)
        info = api.repo_info(REPO, repo_type='dataset')
    commit = api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=str(PACKAGE),
                               commit_message='Add 600 synthetic final manuscripts labeled by last revision model', parent_commit=info.sha)
    receipt = {'repo_id': REPO, 'url': 'https://huggingface.co/datasets/' + REPO, 'private': True,
               'commit': commit.oid, 'status': 'uploaded_verification_pending', 'papers': 600}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    assert api.repo_info(REPO, repo_type='dataset', revision=commit.oid).private
    expected = set(checksums) | {'file_checksums.json', '.gitattributes'}
    assert set(api.list_repo_files(REPO, repo_type='dataset', revision=commit.oid)) == expected
    verified = {}
    for name in sorted(set(checksums) | {'file_checksums.json'}):
        downloaded = hf_hub_download(REPO, name, repo_type='dataset', revision=commit.oid, token=token,
                                     cache_dir='/tmp/pangram-synthetic600-hf-verification')
        data = Path(downloaded).read_bytes()
        assert data == (PACKAGE / name).read_bytes(), name
        verified[name] = hashlib.sha256(data).hexdigest()
        if name.endswith('.parquet'):
            rows = pq.read_table(downloaded).to_pylist()
            assert len(rows) == 600 and [row['paper_id'] for row in rows] == list(range(1, 601))
            assert rows[5]['model'] == 'gpt-6.1-sol' and all(row['model'] == 'gpt-6-luna' for row in rows[:4])
    from datasets import load_dataset
    dataset = load_dataset(REPO, revision=commit.oid, token=token, split='train', streaming=True)
    assert sum(1 for _ in dataset) == 600
    receipt.update(status='complete', verified_files=verified, remote_dataset_rows=600,
                   model_label='model used in last manuscript revision')
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'verified_files'}), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'http_status': getattr(getattr(error, 'response', None), 'status_code', None)}), flush=True)
        raise SystemExit(1)
