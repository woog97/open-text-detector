"""Finish the currently running scale batch, reconcile billing, and publish the viewer."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/data/paper-gap2500-v3-luna-20260930'
RUN = [sys.executable, '-u', str(ROOT / 'benchmarks/otd/paper_scale2500_run.py')]
STATUSES = [OUT / (s + '-status.json') for s in ['outline', 'writer', 'audit']] + [OUT / 'fidelity-review/fidelity-status.json']
started = time.time()
initial = {p: p.read_bytes() if p.exists() else None for p in STATUSES}

def read(path):
    return json.loads(path.read_text())

def command(args):
    subprocess.run(args, cwd=ROOT, check=True)

try:
    # The active runner completes all requests before writing a failed stage status.
    # Its successful path exports scale-report.json only after every stage completes.
    while not (OUT / 'scale-report.json').exists():
        failures = [p for p in STATUSES if p.exists() and p.read_bytes() != initial[p] and read(p)['errors']]
        if failures:
            errors = [e for p in failures for e in read(p)['errors']]
            if not all('retries exhausted' in e for e in errors):
                raise RuntimeError('Runner requires inspection: ' + str(errors))
            time.sleep(3)
            for retry in range(3):
                result = subprocess.run(RUN + ['run'], cwd=ROOT)
                if result.returncode == 0:
                    break
                errors = [e for p in STATUSES if p.exists() for e in read(p)['errors']]
                if not errors or not all('retries exhausted' in e for e in errors):
                    raise RuntimeError('Resume requires inspection: ' + str(errors))
            else:
                raise RuntimeError('Three resume attempts exhausted; outputs retained')
            break
        if time.time() - started > 6 * 3600:
            raise RuntimeError('Completion deadline reached; inspect active runner')
        time.sleep(15)
    # Provider account usage can lag per-response billing records.
    expected = read(OUT / 'scale-report.json')['costs']['new']['account_precision_cost_usd']
    before = read(OUT / 'key-usage-before.json')['usage']
    for _ in range(8):
        command(RUN + ['settle'])
        delta = read(OUT / 'key-total-after.json')['usage'] - before
        if abs(delta - expected) < 1e-8:
            break
        time.sleep(30)
    else:
        raise RuntimeError('Billing has not reconciled; viewer remains unpublished')
    command(RUN + ['export'])
    command(RUN + ['validate'])
    command([sys.executable, '-u', str(ROOT / 'pilot-viewer/build_scale_data.py')])
    result = {'status': 'complete', 'finished_utc_epoch': time.time(), 'costs': read(OUT / 'scale-report.json')['costs']['new'],
              'viewer': 'http://127.0.0.1:3020/#run=gap2500-2020'}
except Exception as exc:
    result = {'status': 'needs_inspection', 'error': str(exc), 'finished_utc_epoch': time.time()}
    raise
finally:
    (OUT / 'completion-status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
