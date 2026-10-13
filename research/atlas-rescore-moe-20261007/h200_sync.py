"""H200 <-> bucket sync for the two-ended Atlas rescore (runs on the H200; HF token from stdin, kept in memory only).

Every --every seconds:
  1. uploads finished H200 shards (out/data/summary-*.parquet + out/tokens/tokens-*.npz pairs) not yet uploaded to
     bucket workspace/atlas-rescore-moe-20261007/out-h200/ (one batch call per round);
  2. lists the Space's finished shards in .../out-space/ and mirrors their names as empty marker files in --space-done,
     so the H200 workers (score_moe.py --done-dir SPACE_DONE) skip them.
Stops when out/ and the Space together cover --total shards, after a final upload.
Usage: echo TOKEN | python h200_sync.py --out DIR --space-done DIR --total 425 [--every 300]
"""
import argparse, json, sys, time
from pathlib import Path
from huggingface_hub import HfApi

B = 'open-text-detector/training-storage'; P = 'workspace/atlas-rescore-moe-20261007'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); ap.add_argument('--space-done', required=True, type=Path)
    ap.add_argument('--total', type=int, required=True); ap.add_argument('--every', type=int, default=300)
    a = ap.parse_args(); api = HfApi(token=sys.stdin.readline().strip()); sent = set(); log = open(a.out.parent / 'sync.log', 'a')
    def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()
    for d in ('data', 'tokens'): (a.space_done / d).mkdir(parents=True, exist_ok=True)
    while True:
        try:
            names = {p.name for p in (a.out / 'data').glob('summary-*.parquet')} if (a.out / 'data').exists() else set()
            ks = sorted(int(n[8:13]) for n in names if (a.out / 'tokens' / f'tokens-{n[8:13]}.npz').exists())
            new = [k for k in ks if k not in sent]
            if new:
                add = [(str(a.out / 'data' / f'summary-{k:05}.parquet'), f'{P}/out-h200/data/summary-{k:05}.parquet') for k in new] + \
                      [(str(a.out / 'tokens' / f'tokens-{k:05}.npz'), f'{P}/out-h200/tokens/tokens-{k:05}.npz') for k in new]
                api.batch_bucket_files(B, add=add); sent |= set(new)
            space = {f.path.rsplit('/', 1)[1][8:13] for f in api.list_bucket_tree(B, prefix=f'{P}/out-space/data/', recursive=True)
                     if f.path.endswith('.parquet')}
            stok = {f.path.rsplit('/', 1)[1][7:12] for f in api.list_bucket_tree(B, prefix=f'{P}/out-space/tokens/', recursive=True)
                    if f.path.endswith('.npz')}
            for s in space & stok:
                (a.space_done / 'data' / f'summary-{s}.parquet').touch(); (a.space_done / 'tokens' / f'tokens-{s}.npz').touch()
            covered = set(ks) | {int(s) for s in space & stok}
            say(event='round', h200=len(ks), uploaded=len(sent), new=len(new), space=len(space & stok), covered=len(covered), total=a.total)
            if len(covered) >= a.total and not [k for k in ks if k not in sent]: say(event='done'); return
        except Exception as e:
            say(event='error', error=repr(e)[:300])
        time.sleep(a.every)


if __name__ == '__main__':
    main()
