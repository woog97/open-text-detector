"""Build Paper Atlas reader details (sentence scores + position maps) for classified papers.

Run on the training Space. For each paper: clean text (positioned-clean-v2 dataset),
sentence scores (sentence_scores_space.py output) and a position map built from the
clean artifact's word mapping and the source extraction's word rectangles. Writes
content-addressed gzipped details to OUT/objects/<sha256>.json.gz and OUT/index.parquet
(pdf_sha256, detail_key, bytes, score_summary_json, mapped). Resumable per shard.
Scores are raw model probabilities; labels use the frozen 1%-FPR sentence threshold.
Usage: build_atlas_details.py --clean DATASET [--clean ...] --sentences DIR --out DIR [--limit N]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

MODEL = {'id': 'qwen35-4b', 'name': 'Qwen3.5-4B · Experimental'}
THRESHOLD = 0.0283355712890625   # sentence threshold at 1% FPR on human papers (results.json)
NOTICE = ('Experimental model (Qwen3.5-4B, trained on 10% of the data). Scores are raw model probabilities. '
          '“Elevated evidence” marks sentences above the threshold that flags 1% of sentences in human-written '
          'papers from 2022 or earlier (215 papers). Flags measure resemblance to AI-generated text and are not '
          'conclusions about authorship. See the Calibration page.')


def read(path, tries=6):
    for attempt in range(tries):
        try: return Path(path).read_bytes()
        except OSError:
            if attempt == tries - 1: raise
            time.sleep(2 * (attempt + 1))


def summarize(segments, text):
    """Same histogram rule as scripts/atlas_scores.py, plus the calibrated flagged count."""
    histogram = [0] * 21; excluded = 0; flagged = 0
    for s in segments:
        if not any(c.isalnum() for c in text[s['start']:s['end']]): excluded += 1; continue
        histogram[math.ceil(s['score'] * 20)] += 1
        flagged += s['label'] == 'ai_evidence'
    return {'histogram': histogram, 'total': sum(histogram), 'excluded': excluded, 'policy': 2,
            'flagged': flagged, 'flag_threshold': THRESHOLD, 'flag_target_fpr': 0.01}


def one(job):
    row, roots, sentences = job
    sys.path.insert(0, roots['backend'])
    from open_text_detector_backend.result_codec import decode
    text = row['text']
    clean = decode(read(Path(roots['clean'][row['clean_dataset']]) / row['clean_positions_path']))
    if clean['text'] != text: raise ValueError('Clean artifact text mismatch')
    source = decode(read(Path(roots['source'][row['source_dataset']]) / row['source_positions_path']))
    if source['text_sha256'] != clean['source_text_sha256']: raise ValueError('Source artifact mismatch')
    rects = source['rectangles']
    position = {'pages': source['pages'],
                'rectangles': [{'start': m['start'], 'end': m['end'], 'page': rects[m['word_index']]['page'],
                                **{k: round(rects[m['word_index']][k], 2) for k in ('x0', 'y0', 'x1', 'y1')}} for m in clean['mapping']]}
    segments = [{'start': int(a), 'end': int(b), 'score': round(float(s), 5), 'label': 'ai_evidence' if float(s) > THRESHOLD else 'below_threshold'}
                for a, b, s in zip(sentences['starts'], sentences['ends'], sentences['scores'].astype(np.float32))]
    detail = {'id': row['pdf_sha256'][:24], 'version': row['pdf_sha256'], 'version_verified': True,
              'reports': [{'id': f"{MODEL['id']}-{row['text_sha256'][:16]}", 'text': text, 'text_sha256': row['text_sha256'],
                           'model': {'name': MODEL['name']}, 'result': {'segments': segments, 'notice': NOTICE}}],
              'position_maps': {row['text_sha256']: position}}
    raw = gzip.compress(json.dumps(detail, separators=(',', ':'), ensure_ascii=False).encode(), compresslevel=6, mtime=0)
    digest = hashlib.sha256(raw).hexdigest()
    path = Path(roots['out']) / 'objects' / f'{digest}.json.gz'
    if not path.exists():
        tmp = path.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_bytes(raw); tmp.replace(path)
    return {'pdf_sha256': row['pdf_sha256'], 'id': row['pdf_sha256'][:24], 'detail_key': f'atlas-public/objects/{digest}.json.gz',
            'bytes': len(raw), 'segments': len(segments), 'rectangles': len(position['rectangles']),
            'score_summary_json': json.dumps(summarize(segments, text))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean', action='append', required=True, type=Path)
    p.add_argument('--sentences', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--backend', required=True)
    p.add_argument('--source-root', required=True, type=Path, help='Directory holding the source extraction datasets')
    p.add_argument('--clean-root', type=Path, help='Directory holding the clean datasets\' objects (default: --clean paths)')
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    import pyarrow as pa, pyarrow.parquet as pq, zipfile
    rows = []
    for d in a.clean:
        for s in sorted((d / 'data').glob('train-*.parquet')):
            for r in pq.read_table(s, columns=['id', 'pdf_sha256', 'text', 'text_sha256', 'source_dataset', 'source_positions_path', 'clean_positions_path']).to_pylist():
                r['clean_dataset'] = d.name; rows.append(r)
    by_id = {r['id']: r for r in rows}
    roots = {'backend': a.backend, 'out': str(a.out),
             'clean': {d.name: str((a.clean_root / d.name) if a.clean_root else d) for d in a.clean},
             'source': {r['source_dataset']: str(a.source_root / r['source_dataset']) for r in rows}}
    (a.out / 'objects').mkdir(parents=True, exist_ok=True); (a.out / 'index').mkdir(exist_ok=True)
    shards = sorted(a.sentences.glob('sentences-*.npz'))
    if a.limit: shards = shards[:max(1, a.limit // 100)]
    started = time.time(); done = 0
    with ProcessPoolExecutor(a.workers) as pool:
        for shard in shards:
            dest = a.out / 'index' / shard.name.replace('sentences-', 'index-').replace('.npz', '.parquet')
            if dest.exists(): continue
            z = np.load(shard); ids = sorted({k.rsplit('/', 1)[0] for k in z.files})
            jobs = [(by_id[i], roots, {k: z[f'{i}/{k}'] for k in ('starts', 'ends', 'scores')}) for i in ids]
            out = []
            for job, fut in zip(jobs, [pool.submit(one, j) for j in jobs]):
                try: out.append(fut.result())
                except Exception as e: out.append({'pdf_sha256': job[0]['pdf_sha256'], 'id': job[0]['pdf_sha256'][:24], 'error': f'{type(e).__name__}: {e}'})
            pq.write_table(pa.Table.from_pylist(out), dest.with_suffix('.tmp')); dest.with_suffix('.tmp').replace(dest)
            done += len(out)
            print(json.dumps({'shard': shard.name, 'papers': done, 'errors': sum('error' in r for r in out), 'seconds': round(time.time() - started)}), flush=True)


if __name__ == '__main__':
    main()
