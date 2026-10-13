import json, shutil, os
from pathlib import Path
R = Path('/tmp/pangram-splice-20261006'); D = Path('/data/workspace/splice-sweep-20261006'); D.mkdir(parents=True, exist_ok=True)
out = {}
for d in sorted((R / 'sweeps').glob('q4b-SP*')):
    t = D / d.name
    shutil.copytree(d, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.tmp', 'snapshots'))
    bad = [str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and 'snapshots' not in p.parts
           and (not (t / p.relative_to(d)).exists() or (t / p.relative_to(d)).stat().st_size != p.stat().st_size)]
    out[d.name] = {'files': sum(1 for p in t.rglob('*') if p.is_file()), 'mismatch': bad[:5],
                   'checkpoints': sorted(p.name for p in t.glob('*.safetensors')), 'pruned': json.loads((d / 'pruned.json').read_text()) if (d / 'pruned.json').exists() else None}
for n in ['setup.log', 'setup_splice.py', 'models.json', 'sweeps/queue.json']:
    shutil.copy(R / n, D / Path(n).name)
for v in ['qwen35-4b-sph', 'qwen35-4b-spg']:
    (D / 'prepared' / v).mkdir(parents=True, exist_ok=True)
    shutil.copytree(R / 'runs' / v / 'prepared-v2', D / 'prepared' / v / 'prepared-v2', dirs_exist_ok=True)
print(json.dumps(out, indent=1))
