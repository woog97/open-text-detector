"""Space-side: build the wide human pool for sampling hosts (paragraphs) and sections from the corpus clean v2 text.

Reads corpus-clean-text-v2 (or --clean-parts for the per-chunk parts while the corpus job runs). Keeps papers dated
2022 or earlier that pass the leakage screens: calibration set, never_train_hit (all held-out lists incl. T2.1 v1/v2),
evaluation/selection/calibration paper ids and shared 10+-word sentences. Splits each paper into host paragraphs
(build_hosted.split_paragraphs + host_ok, the same as the existing hosts) and top-level sections
(build_sections.segment_archive, with top_sections_v2 as fallback). Writes, under --out:
  hosts.parquet      one row per eligible paragraph (host record format of hosts-raw.jsonl, source 'openreview')
  segments.jsonl.gz  one record per segmentable paper (segments.jsonl.gz format, source 'corpus')
  papers.parquet     one row per screened paper: venue, year, counts, reason excluded (if any)
  stats.json         counts by venue/year; this is the only thing worth reading
"""
import argparse, glob, gzip, json, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent / 'repo'
for d in ('research/llm-sentence-edits-20261005', 'research/claude-hosted-edits-20261006', 'research/claude-sections-20261006', 'research/claude-fullpapers-20261006'):
    sys.path.insert(0, str(REPO / d))
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
EVAL = None


def sents(t): return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}


def eval_sets():
    keys, held = set(), set()
    t2 = next(p for p in (Path('/tmp/pangram-t2-20261006'), Path('/data/workspace/pangram-t2-20261006')) if (p / 'sweeps/sweep-eval-rows.jsonl.gz').exists())
    for r in (json.loads(l) for l in gzip.open(t2 / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt')):
        keys.add(r['id'].split('/')[0]); keys.add(str(r.get('group_id') or '').replace('paper:', '')); held |= sents(r['text'])
    base = Path('/data/workspace/pangram-t2-20261006/prepared/qwen35-4b-t2/prepared-v2')
    for name in ['selection-windows', 'calibration-windows']:
        for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
            r = json.loads(l); keys.add(str(r.get('paper_id', '')).replace('paper:', '')); keys.add(r['id'].split('/')[0]); held |= sents(r['text'])
    keys.discard('')
    return keys, held


def clean_paragraphs(text):
    """clean v2 text has one paragraph per line; a line starting lowercase continues the previous one (page breaks)."""
    out = []
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        if out and line[:1].islower() and not re.search(r'[.!?:]["”)]?$', out[-1]): out[-1] += ' ' + line
        else: out.append(line)
    return out


def one(row):
    import build_hosted as bh, build_sections as bs, build_llm_edits as ble
    from build_llm_edits import never_train_hit
    from build_fullpapers import archive_abstract
    ble.paragraphs = clean_paragraphs  # segment_archive imports it from build_llm_edits at call time
    keys, held = EVAL
    sha, text = row['pdf_sha256'], row['text']
    base = {'pdf_sha256': sha, 'openreview_id': row.get('openreview_id') or row.get('id'), 'venue': row.get('conference'), 'year': row.get('year')}
    if {sha, str(row.get('openreview_id')), str(row.get('id'))} & keys: return {**base, 'excluded': 'eval_paper_id'}, [], None
    if never_train_hit({'paper_id': sha, 'id': str(row.get('openreview_id') or ''), 'text': ''}): return {**base, 'excluded': 'never_train_paper'}, [], None
    ps = clean_paragraphs(text); hosts = []; drop = Counter()
    for i, p in enumerate(ps):
        if not bh.host_ok(p): continue
        ctx = '\n\n'.join(x for x in (ps[i - 1] if i else '', p, ps[i + 1] if i + 1 < len(ps) else '') if x)
        if never_train_hit({'paper_id': sha, 'text': ctx}): drop['host_never_train'] += 1; continue
        if sents(ctx) & held: drop['host_eval_sentence'] += 1; continue
        hosts.append({'host_id': f'or:{sha[:16]}/p{i:03d}', 'record_id': sha, 'source': 'openreview', 'year': row.get('year'),
                      'parent_document_id': sha, 'prev': ps[i - 1] if i else '', 'target': p, 'next': ps[i + 1] if i + 1 < len(ps) else '',
                      'venue': row.get('conference'), 'openreview_id': base['openreview_id']})
    seg = bs.segment_archive(text)
    if seg is None:
        orig = bs.top_sections; bs.top_sections = bs.top_sections_v2
        try: seg = bs.segment_archive(text)
        finally: bs.top_sections = orig
    rec = None
    if seg:
        title = (row.get('title') or text.split('\n', 1)[0])[:300]
        rec = {'paper_id': sha, 'forum_id': base['openreview_id'], 'source': 'corpus', 'paired_split': None, 'title': re.sub(r'\s+', ' ', title).strip(),
               'abstract': archive_abstract(text) or '', 'conference': row.get('conference'), 'year': row.get('year'), 'front': seg[0], 'sections': seg[1]}
        rec['eligible'] = []
        for i, r in bs.eligible_sections(rec):
            st = '\n\n'.join(rec['sections'][i]['paragraphs'])
            if never_train_hit({'paper_id': sha, 'text': st}): drop['section_never_train'] += 1; continue
            if sents(st) & held: drop['section_eval_sentence'] += 1; continue
            rec['eligible'].append([i, r])
    return {**base, 'excluded': None, 'hosts': len(hosts), 'segmented': bool(seg), 'eligible_sections': len(rec['eligible']) if rec else 0,
            **{k: v for k, v in drop.items()}}, hosts, rec


def init(e):
    global EVAL; EVAL = e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--clean', default='/data/workspace/datasets/corpus-clean-text-v2')
    ap.add_argument('--clean-parts', help='glob of per-chunk clean.parquet files (while the corpus job runs)')
    ap.add_argument('--calibration', required=True, help='human-calibration selection.json (sha256 list to exclude)')
    ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=80); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import pyarrow as pa, pyarrow.parquet as pq
    files = sorted(glob.glob(a.clean_parts)) if a.clean_parts else sorted(glob.glob(a.clean + '/data/*.parquet'))
    cal = {p['sha256'] for p in json.load(open(a.calibration))['papers']}
    meta = json.loads(Path('/data/workspace/corpus-extraction-20261007/pdf-metadata.json').read_text())
    rows = []
    for f in files:
        for r in pq.read_table(f, columns=['pdf_sha256', 'id', 'title', 'conference', 'year', 'text']).to_pylist():
            m = meta.get(r['pdf_sha256'], {})
            r['conference'] = (r['conference'] or m.get('venue') or '').lower() or None; r['year'] = r['year'] or m.get('year'); r['openreview_id'] = m.get('openreview_id') or r['id']
            if r['year'] and int(r['year']) <= 2022: rows.append(r)
    stats = Counter(candidates=len(rows)); papers = []
    rows = [r for r in rows if not (r['pdf_sha256'] in cal and papers.append({'pdf_sha256': r['pdf_sha256'], 'venue': r['conference'], 'year': r['year'], 'excluded': 'calibration'}) is None)]
    if a.limit: rows = rows[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); hosts = []; segs = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(eval_sets(),)) as pool:
        for p, h, s in pool.map(one, rows, chunksize=8):
            papers.append(p); hosts += h
            if s: segs.append(s)
    cols = ['pdf_sha256', 'openreview_id', 'venue', 'year', 'excluded', 'hosts', 'segmented', 'eligible_sections',
            'host_never_train', 'host_eval_sentence', 'section_never_train', 'section_eval_sentence']
    papers = [{c: p.get(c) for c in cols} for p in papers]
    pq.write_table(pa.Table.from_pylist(hosts), out / 'hosts.parquet'); pq.write_table(pa.Table.from_pylist(papers), out / 'papers.parquet')
    with gzip.open(out / 'segments.jsonl.gz', 'wt') as f:
        for s in segs: f.write(json.dumps(s, ensure_ascii=False) + '\n')
    by = defaultdict(Counter)
    for p in papers:
        g = f"{p['venue']}/{p['year']}"; by[g]['papers'] += 1
        if p['excluded']: by[g]['excluded_' + p['excluded']] += 1; continue
        by[g]['kept'] += 1; by[g]['hosts'] += p['hosts'] or 0; by[g]['segmented'] += bool(p['segmented']); by[g]['eligible_sections'] += p['eligible_sections'] or 0
        by[g]['papers_with_hosts'] += (p['hosts'] or 0) > 0; by[g]['papers_with_eligible_sections'] += (p['eligible_sections'] or 0) > 0
        for k in ('host_never_train', 'host_eval_sentence', 'section_never_train', 'section_eval_sentence'): by[g][k] += p.get(k) or 0
    tot = Counter()
    for c in by.values(): tot.update(c)
    res = {'total': dict(tot), 'by_venue_year': {g: dict(c) for g, c in sorted(by.items())}, 'section_roles': dict(Counter(r for s in segs for _, r in s['eligible']))}
    (out / 'stats.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res['total'], indent=1)); print(json.dumps(res['section_roles']))


if __name__ == '__main__':
    main()
