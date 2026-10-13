"""Publish successive verified archive snapshots until the v8 batch is complete."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/exports/atlas-public'
with (OUT/'publication.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    deadline=time.time()+4*3600
    while time.time()<deadline:
        archive=json.loads((ROOT/'research/exports/classified-r2/summary.json').read_text())
        published=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {}
        available=archive.get('latest_run',{}).get('v8_completed',0)
        if published.get('models',{}).get('v8',-1)<available:
            subprocess.run([sys.executable,str(ROOT/'scripts/publish_verified_atlas.py')],check=True)
            published=json.loads((OUT/'summary.json').read_text())
        if published.get('models',{}).get('v8')==14561:
            print('All 14,561 papers published for both models.',flush=True)
            break
        time.sleep(45)
    else:raise RuntimeError('Publication deadline reached before complete v8 coverage')
