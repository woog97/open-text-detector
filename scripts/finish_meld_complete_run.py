"""Collect completed metadata, release the authorized rental, then verify outputs."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/classifications/vast-complete-20260925'
state = json.loads((RUN / 'run-state.json').read_text())
assert state['instance_id'] == 52521593
summary_path = '/workspace/pangram/research/classifications/vast-complete-20260925/metadata/summary.json'

while time.time() < state['retrieve_at']:
    try:
        probe = subprocess.run(state['ssh'] + ['cat ' + summary_path], capture_output=True, timeout=40)
        summary = json.loads(probe.stdout)
    except (ValueError, TypeError, OSError, subprocess.TimeoutExpired):
        summary = {}
    if summary.get('completed') == {'v5': 7076, 'v8': 14561}:
        break
    time.sleep(45)

subprocess.run([sys.executable, str(ROOT / 'scripts/snapshot_meld_complete.py')], check=True, timeout=150)
counts = {}
for version, expected in [('v5', 7076), ('v8', 14561)]:
    rows = [json.loads(line) for line in (RUN / 'metadata' / (version + '-results.jsonl')).read_text().splitlines()]
    counts[version] = len({row['text_sha256'] for row in rows})
    assert counts[version] == len(rows)
print('Retrieved final metadata:', counts, flush=True)

key = (Path.home() / '.config/vastai/vast_api_key').read_text().strip()
for attempt in range(12):
    try:
        req = urllib.request.Request('https://console.vast.ai/api/v0/instances/52521593/', method='DELETE',
                                     headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.load(response)
        if result.get('success'):
            (RUN / 'run-state.destroyed').write_text(json.dumps({'response': result, 'destroyed_at': time.time()}))
            print('Rental destruction confirmed', flush=True)
            break
    except Exception as exc:
        print('Destruction retry:', type(exc).__name__, flush=True)
    time.sleep(5)
else:
    raise RuntimeError('Rental destruction not confirmed; independent watchdog remains armed')

subprocess.run([sys.executable, str(ROOT / 'scripts/finalize_meld_complete_run.py')], check=True)
print('Finalization complete:', counts, flush=True)
