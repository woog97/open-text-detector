"""Read-only check of the established Hugging Face organization; no uploads."""
import json
from pathlib import Path
from huggingface_hub import HfApi, get_token

root = Path(__file__).resolve().parents[2]
try:
    token = next(line.split('=', 1)[1].strip().strip(chr(34) + chr(39)) for line in (root / 'benchmarks/otd/.env.secrets').read_text().splitlines() if line.startswith('HF_TOKEN='))
    record = {'read_only': True, 'credentials': []}
    for label, credential in [('project', token), ('local_login', get_token())]:
        if not credential:
            continue
        api = HfApi(token=credential)
        who = api.whoami()
        auth = who.get('auth', {}).get('accessToken', {})
        record['credentials'].append({'credential_source': label, 'account': who['name'],
            'token_role': auth.get('role'), 'organizations': who.get('orgs', []),
            'existing_org_datasets': [d.id for d in api.list_datasets(author='open-text-detector')]})
    (root / 'research/data/synthetic-papers-600-hf-destination-check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
except Exception as error:
    print(json.dumps({'error_type': type(error).__name__, 'http_status': getattr(getattr(error, 'response', None), 'status_code', None)}))
    raise SystemExit(1)
