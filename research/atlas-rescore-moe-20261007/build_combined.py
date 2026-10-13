"""Combined Atlas detail files: the live 4B report (unchanged) followed by the final-MoE report, one object per paper.

Runs on the Space after build_staging.py. For each Atlas paper it rebuilds the live 4B detail with
scripts/build_atlas_details.py's rules and constants (model qwen35-4b, threshold 0.0283, its notice) from the live 4B
sentence scores and checks that its content hash equals the live detail_key, i.e. the 4B report is byte-identical to
what the Atlas serves. It then writes {reports: [4B, MoE], position_maps} (both reports share one position map, same
clean text) as a new content-addressed object. Live 4B objects are never modified or deleted.
Outputs /tmp/atlas-rescore/details-combined/{objects/, index/index-xxxxx.parquet, combined-index.parquet}.
Usage: build_combined.py [--workers 48] [--limit N]
"""
import argparse, gzip, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

D = Path('/data/workspace/atlas-rescore-moe-20261007'); L = Path('/tmp/atlas-rescore'); OUT = L / 'details-combined'
DATA = Path('/data/workspace/datasets'); CLEAN = [DATA / 'iclr2027-clean-text-v2-20261005', DATA / 'iclr2027-backfill-clean-text-v2-20261005']
OLD_SENT = Path('/data/workspace/classifications/iclr2027-clean-v2-20261005/qwen35-4b-fast10/sentences'); NEW_SENT = L / 'sentences'
OLD_INDEX = Path('/data/workspace/classifications/iclr2027-clean-v2-20261005/qwen35-4b-fast10/atlas-details-index.parquet')
sys.path.insert(0, str(D / 'code')); import build_atlas_details as b  # noqa: E402
OLD = {'model': dict(b.MODEL), 'threshold': b.THRESHOLD, 'notice': b.NOTICE}
NEW_T = json.loads((D / 'calibration-moe.json').read_text())['variants']['kept+old (as Atlas calibration)']['thresholds'][1]['threshold']
NEW = {'model': {'id': 'qwen36-35b-a3b', 'name': 'Qwen3.6 35B-A3B MoE · Oct 7'}, 'threshold': NEW_T,
       'notice': ('Final model (Qwen3.6-35B-A3B MoE, trained Oct 7, 2026 on the full single-pass mix). Scores are raw model probabilities. '
                  '“Elevated evidence” marks sentences above the threshold that flags 1% of sentences in human-written papers from '
                  '2022 or earlier (2,403 papers). Flags measure resemblance to AI-generated text and are not conclusions about '
                  'authorship. See the Calibration page.')}


def report(spec, row, sent):
    text = row['text']
    segs = [{'start': int(a), 'end': int(e), 'score': round(float(s), 5), 'label': 'ai_evidence' if float(s) > spec['threshold'] else 'below_threshold'}
            for a, e, s in zip(sent['starts'], sent['ends'], sent['scores'].astype(np.float32))]
    rep = {'id': f"{spec['model']['id']}-{row['text_sha256'][:16]}", 'text': text, 'text_sha256': row['text_sha256'],
           'model': {'name': spec['model']['name']}, 'result': {'segments': segs, 'notice': spec['notice']}}
    hist = [0] * 21; excl = 0; flagged = 0
    for s in segs:
        if not any(c.isalnum() for c in text[s['start']:s['end']]): excl += 1; continue
        hist[int(np.ceil(s['score'] * 20))] += 1; flagged += s['label'] == 'ai_evidence'
    return rep, {'histogram': hist, 'total': sum(hist), 'excluded': excl, 'policy': 2, 'flagged': flagged, 'flag_threshold': spec['threshold'], 'flag_target_fpr': 0.01}


def gz(detail): return gzip.compress(json.dumps(detail, separators=(',', ':'), ensure_ascii=False).encode(), compresslevel=6, mtime=0)


def one(job):
    row, roots, old_s, new_s, live_key = job
    sys.path.insert(0, roots['backend'])
    from open_text_detector_backend.result_codec import decode
    clean = decode(b.read(Path(roots['clean'][row['clean_dataset']]) / row['clean_positions_path']))
    if clean['text'] != row['text']: raise ValueError('Clean artifact text mismatch')
    source = decode(b.read(Path(roots['source'][row['source_dataset']]) / row['source_positions_path']))
    if source['text_sha256'] != clean['source_text_sha256']: raise ValueError('Source artifact mismatch')
    rects = source['rectangles']
    position = {'pages': source['pages'], 'rectangles': [{'start': m['start'], 'end': m['end'], 'page': rects[m['word_index']]['page'],
                **{k: round(rects[m['word_index']][k], 2) for k in ('x0', 'y0', 'x1', 'y1')}} for m in clean['mapping']]}
    base = {'id': row['pdf_sha256'][:24], 'version': row['pdf_sha256'], 'version_verified': True}
    old_rep, old_sum = report(OLD, row, old_s); new_rep, new_sum = report(NEW, row, new_s)
    old_raw = gz({**base, 'reports': [old_rep], 'position_maps': {row['text_sha256']: position}})
    old_ok = f'atlas-public/objects/{hashlib.sha256(old_raw).hexdigest()}.json.gz' == live_key
    raw = gz({**base, 'reports': [old_rep, new_rep], 'position_maps': {row['text_sha256']: position}})
    digest = hashlib.sha256(raw).hexdigest(); path = Path(roots['out']) / 'objects' / f'{digest}.json.gz'
    if not path.exists():
        tmp = path.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_bytes(raw); tmp.replace(path)
    return {'pdf_sha256': row['pdf_sha256'], 'id': row['pdf_sha256'][:24], 'detail_key': f'atlas-public/objects/{digest}.json.gz', 'bytes': len(raw),
            'live_4b_key': live_key, 'live_4b_identical': old_ok, 'summary_4b_json': json.dumps(old_sum), 'summary_moe_json': json.dumps(new_sum)}


def load_npz_dir(d):
    out = {}
    for f in sorted(d.glob('sentences-*.npz')):
        z = np.load(f)
        for k in z.files:
            pid, kind = k.rsplit('/', 1); out.setdefault(pid, {})[kind] = z[k]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=48); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    import pyarrow as pa, pyarrow.parquet as pq
    rows = []
    for d in CLEAN:
        for s in sorted((d / 'data').glob('train-*.parquet')):
            for r in pq.read_table(s, columns=['id', 'pdf_sha256', 'text', 'text_sha256', 'source_dataset', 'source_positions_path', 'clean_positions_path']).to_pylist():
                r['clean_dataset'] = d.name; rows.append(r)
    live = {r['pdf_sha256']: r['detail_key'] for r in pq.read_table(OLD_INDEX, columns=['pdf_sha256', 'detail_key']).to_pylist() if r['detail_key']}
    old, new = load_npz_dir(OLD_SENT), load_npz_dir(NEW_SENT)
    rows = sorted((r for r in rows if r['id'] in old and r['id'] in new and r['pdf_sha256'] in live), key=lambda r: r['id'])
    if a.limit: rows = rows[:a.limit]
    roots = {'backend': '/tmp/pangram-corpus/code/backend', 'out': str(OUT), 'clean': {d.name: str(d) for d in CLEAN},
             'source': {r['source_dataset']: str(DATA / r['source_dataset']) for r in rows}}
    (OUT / 'objects').mkdir(parents=True, exist_ok=True); (OUT / 'index').mkdir(exist_ok=True); t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for k in range(0, len(rows), 1000):
            dest = OUT / 'index' / f'index-{k // 1000:05d}.parquet'
            if dest.exists(): continue
            part = rows[k:k + 1000]
            jobs = [(r, roots, {x: old[r['id']][x] for x in ('starts', 'ends', 'scores')}, {x: new[r['id']][x] for x in ('starts', 'ends', 'scores')},
                     live[r['pdf_sha256']]) for r in part]
            out = []
            for job, fut in zip(jobs, [pool.submit(one, j) for j in jobs]):
                try: out.append(fut.result())
                except Exception as e: out.append({'pdf_sha256': job[0]['pdf_sha256'], 'id': job[0]['pdf_sha256'][:24], 'error': f'{type(e).__name__}: {e}'})
            pq.write_table(pa.Table.from_pylist(out), dest.with_suffix('.tmp')); dest.with_suffix('.tmp').replace(dest)
            print(json.dumps({'papers': k + len(part), 'of': len(rows), 'errors': sum('error' in r for r in out),
                              'live_4b_identical': sum(bool(r.get('live_4b_identical')) for r in out), 'seconds': round(time.time() - t0)}), flush=True)
    allr = [r for f in sorted((OUT / 'index').glob('index-*.parquet')) for r in pq.read_table(f).to_pylist()]
    pq.write_table(pa.Table.from_pylist(allr), OUT / 'combined-index.parquet')
    print(json.dumps({'done': len(allr), 'errors': sum(bool(r.get('error')) for r in allr), 'live_4b_identical': sum(bool(r.get('live_4b_identical')) for r in allr)}), flush=True)


if __name__ == '__main__':
    main()
