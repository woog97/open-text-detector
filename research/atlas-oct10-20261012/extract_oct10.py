"""Cutoff-explorer ICLR files for the Oct 10 final MoE (qwen36-35b-a3b-oct10), from the staged Oct 10 detail objects.

Copy of research/cutoff-explorer-20261009/extract.py (main checkout, untracked) with three changes:
  - iclr: reads the report whose id starts with 'qwen36-35b-a3b-oct10-' and maps detail objects to PDFs through
    build_oct10.py's index.parquet (detail_key -> pdf_sha256) instead of atlas-details-index-moe.parquet;
  - human: removed (the Oct 10 human-*.parquet files were built separately and are copied into OUT_DIR);
  - finish: the published-number checks are replaced by Oct 10 checks: each mode's human 1% cutoff on all 2,403
    papers must equal recal-final-moe.json's '<mode>/before' threshold (nobib: 0.001318), and the ICLR flagged counts
    at recal's thresholds (score >= cutoff, as the site counts) must equal build_oct10.py's flags-oct10.json.
Usage: extract_oct10.py iclr INDEX.parquet CATALOGUE.json OUT_DIR TAR [TAR ...]
       extract_oct10.py finish RECAL.json FLAGS_OCT10.json OUT_DIR
"""
import gzip, json, sys, tarfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008')); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import cal_analyze as ca  # noqa: E402
PREFIX = 'qwen36-35b-a3b-oct10-'
from bib import MODES, included, section_of, sections  # noqa: E402

SEC = {'body': 0, 'bib': 1, 'app': 2}


def iclr(index, catalogue, out, tars):
    out = Path(out) / 'parts'; out.mkdir(parents=True, exist_ok=True)
    by_pdf = {p['pdf_key'][7:-4]: p for p in json.loads(Path(catalogue).read_text())['items']}
    staged = {r['detail_key']: r['pdf_sha256'] for r in pq.read_table(index, columns=['detail_key', 'pdf_sha256']).to_pylist()}
    by_key = {k: by_pdf[h] for k, h in staged.items() if h in by_pdf}
    for tar in tars:
        part = out / (Path(tar).stem + '.parquet')
        if part.exists(): print(f'{part.name} exists, skipped'); continue
        papers, sents = [], {'id': [], 'section': [], 'score': [], 'start': [], 'end': []}
        with tarfile.open(tar) as tf:
            for m in tf:
                if not m.isfile(): continue
                paper = by_key.get('atlas-public/' + m.name)
                if paper is None: continue
                d = json.loads(gzip.decompress(tf.extractfile(m).read()))
                r = next(x for x in d['reports'] if x['id'].startswith(PREFIX)); t = r['text']; spans, tail = sections(t); n = 0
                for s in r['result']['segments']:
                    if not any(c.isalnum() for c in t[s['start']:s['end']]): continue
                    sents['id'].append(paper['id']); sents['section'].append(SEC[section_of(s['start'], spans, tail)])
                    sents['score'].append(s['score']); sents['start'].append(s['start']); sents['end'].append(s['end']); n += 1
                papers.append({'id': paper['id'], 'title': paper.get('title') or '', 'openreview': (paper.get('filename') or '')[:-4],
                               'collection': paper.get('collection') or '', 'sentences': n, 'text': t})
        tmp = part.with_suffix('.tmp')
        with pq.ParquetWriter(tmp, pa.schema([('id', pa.string()), ('section', pa.uint8()), ('score', pa.float64()), ('start', pa.int32()), ('end', pa.int32())]), compression='zstd') as w:
            w.write_table(pa.table(sents, schema=w.schema))
        pq.write_table(pa.Table.from_pylist(papers), part.with_name(Path(tar).stem + '-papers.parquet'), compression='zstd')
        tmp.rename(part); print(f'{Path(tar).name}: {len(papers):,} papers, {len(sents["score"]):,} sentences', flush=True)


def finish(recal, flags_json, out):
    out = Path(out); parts = out / 'parts'
    papers = pa.concat_tables([(lambda t: t.set_column(t.schema.get_field_index('text'), 'text', t['text'].cast(pa.large_string())))(pq.read_table(f))
                             for f in sorted(parts.glob('*-papers.parquet'))]).sort_by('id')
    ids = papers['id'].to_pylist(); assert len(ids) == len(set(ids)), 'duplicate papers'
    index = {p: i for i, p in enumerate(ids)}
    pq.write_table(papers.select(['id', 'text']).append_column('paper', pa.array(range(len(ids)), pa.int32())), out / 'iclr-text.parquet', compression='zstd', row_group_size=256)
    meta = papers.drop(['text']).append_column('paper', pa.array(range(len(ids)), pa.int32())); pq.write_table(meta, out / 'iclr-papers.parquet')
    s = pa.concat_tables([pq.read_table(f) for f in sorted(parts.glob('objects-*[0-9].parquet'))])
    paper = pa.array([index[x] for x in s['id'].to_pylist()], pa.int32())
    s = s.drop(['id']).append_column('paper', paper).sort_by([('paper', 'ascending'), ('start', 'ascending')])
    pq.write_table(s, out / 'iclr-sentences.parquet', compression='zstd', row_group_size=4_000_000)
    print(f'iclr: {len(ids):,} papers, {s.num_rows:,} sentences')
    # Oct 10 checks.
    recal = json.loads(Path(recal).read_text())['modes']; flags = json.loads(Path(flags_json).read_text())
    h = pq.read_table(out / 'human-sentences.parquet').to_pydict(); hs, hsec = np.array(h['score']), np.array(h['section'])
    iclr_sec, iclr_score, iclr_paper = s['section'].to_numpy(), s['score'].to_numpy(), s['paper'].to_numpy()
    for m in MODES:
        keep = np.isin(hsec, [SEC[x] for x in SEC if included(x, m)]); t = ca.threshold_at(hs[keep], 0.01)
        t_before = next(x['threshold'] for x in recal[m + '/before']['thresholds'] if x['target_fpr'] == 0.01)
        ki = np.isin(iclr_sec, [SEC[x] for x in SEC if included(x, m)]); bad = 0
        for suffix in ('', '0.1'):
            tm = flags['thresholds'][m + suffix]
            flagged = np.bincount(iclr_paper[ki & (iclr_score >= tm)], minlength=len(ids)); total = np.bincount(iclr_paper[ki], minlength=len(ids))
            bad += sum((int(flagged[index[p]]), int(total[index[p]])) != tuple(v['modes'][m + suffix]) for p, v in flags['papers'].items() if p in index)
        print(f'{m}: human 1% cutoff (2,403 papers) {t:.10f} vs recal {m}/before {t_before:.10f} ({"match" if abs(t - t_before) < 1e-12 else "MISMATCH"}); '
              f'ICLR paper-mode counts differing from flags-oct10.json: {bad} (1% and 0.1%) of {len(flags["papers"]):,} papers')

if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'iclr': iclr(a[0], a[1], a[2], a[3:])
    else: finish(*a)
