"""Per-sentence scores of the final MoE for the cutoff explorer (app.py): every ICLR 2027 Atlas paper and the 2,403 human
calibration papers, with each sentence's section (body, bibliography, appendix) so all four counting modes work.

Uses the same sentence rules as research/atlas-bib-exclusion-20261008/recompute.py, which produced the published
thresholds: ICLR sentences are the MoE report's segments in the staged detail objects (segments without letters or
digits dropped); human sentences come from the stored token scores (cal_analyze SENT rule). A sentence is flagged when
score > cutoff. Checks: the 1% cutoff in each mode reproduces calibration-modes.json, and the ICLR flagged counts at
the 1% cutoff reproduce recompute.py atlas (modes-*.json).

Usage: extract.py human WORK OUT_DIR
       extract.py iclr WORK CATALOGUE.json OUT_DIR TAR [TAR ...]   (one part file per tar; resumable)
       extract.py finish WORK OUT_DIR
WORK is the recompute.py work directory (out-cal/, cal-text/, old-text/, overlap.json, atlas-details-index-moe.parquet,
calibration-modes.json, modes-*.json). OUT_DIR gets human-*.parquet, iclr-sentences.parquet, iclr-papers.parquet,
iclr-text.parquet.
"""
import gzip, json, sys, tarfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'atlas-bib-exclusion-20261008')); sys.path.insert(0, str(HERE.parent / 'atlas-rescore-moe-20261007'))
import cal_analyze as ca  # noqa: E402
from bib import MODES, included, section_of, sections  # noqa: E402
from recompute import sentence_scores_spans  # noqa: E402

SEC = {'body': 0, 'bib': 1, 'app': 2}


def human(work, out):
    work, out = Path(work), Path(out); tok, _ = ca.load_tokens(work / 'out-cal'); texts = {}
    for d in ('cal-text', 'old-text'):
        for f in sorted((work / d / 'data').glob('*.parquet')):
            for r in pq.read_table(f, columns=['id', 'text']).to_pylist(): texts[r['id']] = r['text']
    overlap = json.loads((work / 'overlap.json').read_text())['papers']
    keep = {p for p, v in overlap.items() if v['shared_unique'] <= 0 and not v['key_hit']}
    old_ids = {r['id'] for f in sorted((work / 'old-text' / 'data').glob('*.parquet')) for r in pq.read_table(f, columns=['id']).to_pylist()}
    papers, rows = [], {'paper': [], 'section': [], 'score': []}
    for pid, d in sorted(tok.items()):
        if not (pid in keep or pid in old_ids): continue
        s, starts = sentence_scores_spans(texts[pid], d['offsets'], d['probs']); spans, tail = sections(texts[pid])
        key = 'old:' + pid if pid in old_ids else pid; i = len(papers)
        papers.append({'paper': i, 'key': key, 'half': ca.half(key), 'sentences': len(s)})
        rows['paper'] += [i] * len(s); rows['section'] += [SEC[section_of(a, spans, tail)] for a in starts]; rows['score'] += list(s)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(pa.Table.from_pylist(papers).to_pydict()), out / 'human-papers.parquet')
    pq.write_table(pa.table({'paper': pa.array(rows['paper'], pa.int32()), 'section': pa.array(rows['section'], pa.uint8()),
                             'score': pa.array(rows['score'], pa.float64())}), out / 'human-sentences.parquet', compression='zstd')
    print(f'human: {len(papers):,} papers, {len(rows["score"]):,} sentences')


def iclr(work, catalogue, out, tars):
    out = Path(out) / 'parts'; out.mkdir(parents=True, exist_ok=True)
    by_pdf = {p['pdf_key'][7:-4]: p for p in json.loads(Path(catalogue).read_text())['items']}
    staged = {r['detail_key']: r['pdf_sha256'] for r in pq.read_table(Path(work) / 'atlas-details-index-moe.parquet', columns=['detail_key', 'pdf_sha256']).to_pylist()}
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
                r = next(x for x in d['reports'] if x['id'].startswith('qwen36-35b-a3b-')); t = r['text']; spans, tail = sections(t); n = 0
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


def finish(work, out):
    work, out = Path(work), Path(out); parts = out / 'parts'
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
    # Checks against the published numbers.
    cal = json.loads((work / 'calibration-modes.json').read_text())['modes']
    h = pq.read_table(out / 'human-sentences.parquet').to_pydict(); hs, hsec = np.array(h['score']), np.array(h['section'])
    iclr_sec, iclr_score, iclr_paper = s['section'].to_numpy(), s['score'].to_numpy(), s['paper'].to_numpy()
    pub = {}
    for f in sorted(work.glob('modes-*.json')): pub.update(json.loads(f.read_text())['papers'])
    for m in MODES:
        keep = np.isin(hsec, [SEC[x] for x in SEC if included(x, m)]); t = ca.threshold_at(hs[keep], 0.01)
        t_pub = next(x['threshold'] for x in cal[m]['thresholds'] if x['target_fpr'] == 0.01)
        ki = np.isin(iclr_sec, [SEC[x] for x in SEC if included(x, m)])
        flagged = np.bincount(iclr_paper[ki & (iclr_score > t_pub)], minlength=len(ids)); total = np.bincount(iclr_paper[ki], minlength=len(ids))
        bad = sum((int(flagged[index[p]]), int(total[index[p]])) != tuple(v['modes'][m]) for p, v in pub.items() if p in index)
        print(f'{m}: 1% cutoff {t:.10f} vs published {t_pub:.10f} ({"match" if abs(t - t_pub) < 1e-12 else "MISMATCH"}); '
              f'ICLR papers with counts differing from recompute.py: {bad} of {len(pub):,}')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'human': human(*a)
    elif cmd == 'iclr': iclr(a[0], a[1], a[2], a[3:])
    else: finish(*a)
