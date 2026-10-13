"""Merge a prepared-v2 mix's three disjoint stage-2 files into one stage2-epoch0 (Oct 8: stage 2 is ONE epoch; the trainer
shuffles it). Stage 1, selection and calibration are unchanged. Checks: merged rows = sum of the three files, ids unique.
The old epoch1/2 files are renamed *.merged-into-epoch0 (not deleted). Usage: python merge_epochs.py PREPARED_V2_DIR"""
import gzip, hashlib, json, os, sys
from collections import Counter
from pathlib import Path
P = Path(sys.argv[1]); man = json.loads((P / 'manifest.json').read_text())
keys = [f'stage2-epoch{e}' for e in range(3) if f'stage2-epoch{e}' in man['files']]
if keys == ['stage2-epoch0'] and man.get('merged_stage2'):
    print(json.dumps({'already_merged': True, 'rows': man['files']['stage2-epoch0']['rows']})); sys.exit()
lines, n_each = [], {}
for k in keys:
    b = gzip.decompress((P / f'{k}.jsonl.gz').read_bytes()); assert hashlib.sha256(b).hexdigest() == man['files'][k]['sha256'], k
    ls = b.splitlines(); n_each[k] = len(ls); lines += ls
ids = [json.loads(l)['id'] for l in lines]
assert len(lines) == sum(n_each.values()) and len(set(ids)) == len(ids), ('duplicate ids', len(ids) - len(set(ids)))
b = b'\n'.join(lines) + b'\n'; tmp = P / 'stage2-epoch0.jsonl.gz.tmp'; tmp.write_bytes(gzip.compress(b, compresslevel=6))
counts = Counter(json.loads(l)['dataset'] for l in lines)
for k in keys[1:]:
    os.replace(P / f'{k}.jsonl.gz', P / f'{k}.jsonl.gz.merged-into-epoch0'); man['files'].pop(k); man['counts'].pop(k, None)
os.replace(tmp, P / 'stage2-epoch0.jsonl.gz')
man['files']['stage2-epoch0'] = {**man['files']['stage2-epoch0'], 'rows': len(lines), 'sha256': hashlib.sha256(b).hexdigest()}
man['counts']['stage2-epoch0'] = dict(counts); man['merged_stage2'] = {'from': n_each, 'unique_ids': len(ids)}
(P / 'manifest.json').write_text(json.dumps(man, indent=1))
print(json.dumps({'rows': len(lines), 'from': n_each, 'unique_ids': len(set(ids)), 'datasets': len(counts)}))
