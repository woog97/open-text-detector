"""Stage the MoE Atlas details (runs on the Space after scoring; nothing is uploaded to the live Atlas).

1. combine: symlink every Atlas token shard from out-space/ and out-h200/ into run/tokens (one per shard number).
2. sentences: research/human-calibration-20261006/sentence_scores_space.py papers mode -> sentences/ (same rule as live).
3. details: scripts/build_atlas_details.py with the MoE model id, the calibrated threshold and a matching notice, into
   /tmp/atlas-rescore/details-moe (objects + index); then a combined-index.parquet and a tar of the index to the bucket.
Usage: build_staging.py STEP [STEP ...]   (combine | sentences | details | index)
"""
import json, os, subprocess, sys
from pathlib import Path

D = Path('/data/workspace/atlas-rescore-moe-20261007'); L = Path('/tmp/atlas-rescore'); RUN = L / 'run'; SENT = L / 'sentences'; OUT = L / 'details-moe'
DATA = Path('/data/workspace/datasets'); CLEAN = [DATA / 'iclr2027-clean-text-v2-20261005', DATA / 'iclr2027-backfill-clean-text-v2-20261005']
THRESHOLD = json.loads((D / 'calibration-moe.json').read_text())['variants']['kept+old (as Atlas calibration)']['thresholds'][1]['threshold']
CODE = D / 'code'


def combine():
    (RUN / 'tokens').mkdir(parents=True, exist_ok=True); n = 0
    for src in ('out-space', 'out-h200'):
        for f in sorted((D / src / 'tokens').glob('tokens-*.npz')):
            dst = RUN / 'tokens' / f.name
            if not dst.exists(): dst.symlink_to(f); n += 1
    print('linked', n, 'total', len(list((RUN / 'tokens').glob('*.npz'))))


def sentences():
    subprocess.run([sys.executable, '-u', str(CODE / 'sentence_scores_space.py'), 'papers', *map(str, CLEAN), '--run', str(RUN), '--out', str(SENT),
                    '--workers', '48'], check=True)


def details():
    sys.path.insert(0, str(CODE)); import build_atlas_details as b
    b.MODEL = {'id': 'qwen36-35b-a3b', 'name': 'Qwen3.6-35B-A3B MoE · Experimental'}
    b.THRESHOLD = THRESHOLD
    b.NOTICE = ('Experimental model (Qwen3.6-35B-A3B MoE, final run of Oct 7, 2026). Scores are raw model probabilities. '
                '“Elevated evidence” marks sentences above the threshold that flags 1% of sentences in human-written '
                'papers from 2022 or earlier (2,403 papers). Flags measure resemblance to AI-generated text and are not '
                'conclusions about authorship. See the Calibration page.')
    sys.argv = ['build_atlas_details.py', *sum([['--clean', str(c)] for c in CLEAN], []), '--sentences', str(SENT), '--out', str(OUT),
                '--backend', '/tmp/pangram-corpus/code/backend', '--source-root', str(DATA), '--workers', '48']
    b.main()


def index():
    import pyarrow as pa, pyarrow.parquet as pq
    rows = [r for f in sorted((OUT / 'index').glob('index-*.parquet')) for r in pq.read_table(f).to_pylist()]
    pq.write_table(pa.Table.from_pylist(rows), OUT / 'combined-index.parquet')
    (D / 'staging').mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), D / 'staging' / 'atlas-details-index-moe.parquet')
    print('index rows', len(rows), 'errors', sum(1 for r in rows if r.get('error')))
    # Persist the staged detail objects as ~1.5 GB tar shards (no per-file uploads).
    import tarfile
    objs = sorted((OUT / 'objects').glob('*.json.gz')); shard = None; size = 0; n = 0
    for f in objs:
        if shard is None or size > 1.5e9:
            if shard: shard.close()
            shard = tarfile.open(D / 'staging' / f'objects-{n:03}.tar', 'w'); n += 1; size = 0
        shard.add(f, arcname='objects/' + f.name); size += f.stat().st_size
    if shard: shard.close()
    print('object shards', n, 'objects', len(objs))


if __name__ == '__main__':
    for step in sys.argv[1:]: globals()[step]()
