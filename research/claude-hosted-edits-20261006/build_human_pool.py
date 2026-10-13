"""Human scientific-text windows for training (no generation), for the T2 scale-up.

Sources, in order:
  1. Pre-2022 archive papers (open-text-detector/paper-text batch-001; ICLR 2020-22, CoRL 2021-22, NeurIPS 2022,
     TMLR 2022). Excluded: NeurIPS 2021, papers on any never-train list, sweep-eval/selection/calibration/space-suite-v1
     papers, archive papers reserved for held-out sets (never used anywhere), and every paragraph that already appears in
     any AI-edit, section or paragraph-edit row (as the edited text or as context).
  2. Human-source-mix scientific passages (pmc, pes2o, arxiv) with no host used by any job. NOTE: these passages are
     already part of the T2 human pool (pool-human, 75,548 rows from human-source-mix-v1), so they are written to a
     separate file and flagged; use them only if the T2 builder dedupes against its human pool.
Windows are consecutive prose paragraphs packed up to MAX_CHARS characters (~4 characters per token for scientific
English, so <=~475 tokens; the tokenizer is not run locally because model assets stay on the Space/H200). A paragraph
longer than MAX_CHARS is cropped at a sentence boundary. Training prep crops to 510 tokens in any case.

Usage: build_human_pool.py ARCHIVE_PARQUET HSM_DATA_DIR EXCLUDE_IDS_JSON [--overlap HITS_JSON]
"""
import argparse, os, gzip, hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder of the Oct 6 session that made the intermediate files (outside the repo); set SCRATCH to it

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/llm-sentence-edits-20261005'))
sys.path.insert(0, str(HERE))
MAX_CHARS, MIN_CHARS, TARGET = 1900, 600, 5200
SCR = (SCRATCH / 'hsm')


def nh(t):
    return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()


def used_paragraph_hashes():
    hs = set()
    files = [ROOT / 'research/llm-sentence-edits-20261005' / f for f in ('llm-edits-pilot.jsonl.gz', 'llm-edits-v1.jsonl.gz', 'llm-edits-claude-v1.jsonl.gz', 'heldout-eval-v1.jsonl.gz')]
    files += [ROOT / 'research/splice-edits-20261005/splices-v1.jsonl.gz']
    files += list((ROOT / 'research/claude-sections-20261006').glob('claude-sections-v1-*.jsonl.gz')) + list(HERE.glob('*-claude-hsm-v1*.jsonl.gz'))
    for f in files:
        if f.exists():
            for l in gzip.open(f, 'rt'):
                for p in json.loads(l)['text'].split('\n\n'):
                    hs.add(nh(p))
    for f in ('candidates-claude-v1.jsonl', 'candidates-heldout.jsonl', 'candidates-v1.jsonl'):  # assigned but maybe not yet written
        for l in open(ROOT / 'research/llm-sentence-edits-20261005' / f):
            c = json.loads(l)
            for k in ('before', 'target', 'after'):
                hs.add(nh(c[k]))
    sec = ROOT / 'research/claude-sections-20261006'
    items = [json.loads(l) for f in ('items.jsonl', 'items-gap.jsonl', 'items-scale.jsonl') if (sec / f).exists() for l in open(sec / f)]
    need = {it['paper_id'] for it in items}
    sys.path.insert(0, str(sec))
    from build_sections import excerpt
    segs = {}
    for l in gzip.open(sec / 'segments.jsonl.gz', 'rt'):
        x = json.loads(l)
        if x['paper_id'] in need:
            segs[x['paper_id']] = x
    for it in items:  # the section item's excerpt (target section + ~400 words of context each side) is used text
        for p in excerpt(segs[it['paper_id']], it['si'])[0]:
            hs.add(nh(p))
    return hs


def crop(p):
    from build_llm_edits import sentences
    out = ''
    for a, b in sentences(p):
        if len(p[:b]) > MAX_CHARS:
            break
        out = p[:b]
    return out.strip()


def archive_windows(parquet, exclude_json):
    import pyarrow.parquet as pq
    from build_llm_edits import paragraphs, never_train_hit, mathy, reference_like
    sys.path.insert(0, str(ROOT / 'research/claude-sections-20261006'))
    from build_sections import used_archive_ids, prose_ok
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids']); used_arch = used_archive_ids(); used = USED
    stats = Counter(); wins = []
    for r in pq.read_table(parquet, columns=['paper_id', 'forum_id', 'title', 'conference', 'year', 'text']).to_pylist():
        if r['year'] > 2022 or (r['conference'] == 'neurips' and r['year'] == 2021):
            continue
        if r['paper_id'] in ex or r['paper_id'] not in used_arch:
            stats['paper_excluded_eval_or_heldout_reserve'] += 1; continue
        if never_train_hit({'paper_id': r['paper_id'], 'text': ''}):
            stats['paper_never_train'] += 1; continue
        cur, n = [], 0
        def flush():
            if cur and sum(len(x) for x in cur) >= MIN_CHARS:
                wins.append({'paper_id': r['paper_id'], 'source': 'archive_pre2023', 'conference': r['conference'], 'year': r['year'],
                             'text': '\n\n'.join(cur)})
            cur.clear()
        for p in paragraphs(r['text']):
            ok = len(p) >= 200 and prose_ok(p) and not mathy(p) and not reference_like(p) and re.search(r'[.!?]["”)]?$', p)
            if not ok or nh(p) in used:
                stats['para_skipped_used' if ok else 'para_skipped_nonprose'] += 1; flush(); continue
            if len(p) > MAX_CHARS:
                flush(); c = crop(p)
                if len(c) >= MIN_CHARS:
                    cur.append(c); flush()
                continue
            if sum(len(x) + 2 for x in cur) + len(p) > MAX_CHARS:
                flush()
            cur.append(p)
        flush()
    for w in wins:
        if never_train_hit(w):
            w['drop'] = True; stats['window_never_train'] += 1
    wins = [w for w in wins if not w.get('drop')]
    stats['windows'] = len(wins); stats['papers'] = len({w['paper_id'] for w in wins})
    return wins, stats


def paired_windows(exclude_json, used):
    """Paired-data TRAIN papers (ACL/ICML/NeurIPS 2013-2021, official pre-2022 PDFs), body paragraphs from the section
    segmentation, minus every paragraph already used in an AI row."""
    from build_llm_edits import never_train_hit, mathy, reference_like
    sys.path.insert(0, str(ROOT / 'research/claude-sections-20261006'))
    from build_sections import prose_ok
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids']); stats = Counter(); wins = []
    for l in gzip.open(ROOT / 'research/claude-sections-20261006/segments.jsonl.gz', 'rt'):
        x = json.loads(l)
        if x['source'] != 'paired' or x['paired_split'] != 'train' or x['paper_id'] in ex or never_train_hit({'paper_id': x['paper_id'], 'text': ''}):
            continue
        for sec in x['sections']:
            cur = []
            def flush():
                if cur and sum(len(q) for q in cur) >= MIN_CHARS:
                    wins.append({'paper_id': x['paper_id'], 'source': 'paired_train_pre2022', 'conference': x['conference'], 'year': x['year'], 'text': '\n\n'.join(cur)})
                cur.clear()
            for p in sec['paragraphs']:
                ok = len(p) >= 150 and prose_ok(p) and not mathy(p) and not reference_like(p) and re.search(r'[.!?]["”)]?$', p)
                if not ok or nh(p) in used:
                    stats['para_skipped_used' if ok else 'para_skipped_nonprose'] += 1; flush(); continue
                if len(p) > MAX_CHARS:
                    flush(); c = crop(p)
                    if len(c) >= MIN_CHARS:
                        cur.append(c); flush()
                    continue
                if sum(len(q) + 2 for q in cur) + len(p) > MAX_CHARS:
                    flush()
                cur.append(p)
            flush()
    wins = [w for w in wins if not never_train_hit(w)]
    stats['windows'] = len(wins); stats['papers'] = len({w['paper_id'] for w in wins})
    return wins, stats


def hsm_windows(data_dir, exclude_json):
    import pyarrow.parquet as pq
    from build_llm_edits import never_train_hit
    from build_hosted import split_paragraphs, host_ok, relaxed_ok
    assigned = set()
    for f in HERE.glob('items-*.jsonl'):
        if 'unrun' in f.name:
            continue
        for l in open(f):
            assigned.add(json.loads(l)['record_id'])
    held = {json.loads(l)['record_id'] for l in open(HERE / 'items-e.unrun.jsonl') if json.loads(l)['split'] == 'heldout'}
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids']); stats = Counter(); wins = []
    for s_ in ('pmc', 'pes2o', 'arxiv'):
        for r in pq.read_table(Path(data_dir) / 'data' / f'{s_}.parquet', columns=['record_id', 'text', 'claimed_original_date']).to_pylist():
            y = re.match(r'(\d{4})', str(r['claimed_original_date'] or ''))
            if (y and int(y.group(1)) > 2022) or r['record_id'] in assigned or r['record_id'] in held or r['record_id'] in ex:
                continue
            ps = [p for p in split_paragraphs(r['text']) if relaxed_ok(p)]
            text, cur = '', []
            for p in ps:
                if sum(len(x) + 2 for x in cur) + len(p) > MAX_CHARS:
                    break
                cur.append(p)
            text = '\n\n'.join(cur)
            if len(text) < MIN_CHARS:
                stats['too_short'] += 1; continue
            w = {'paper_id': r['record_id'], 'source': f'hsm_{s_}', 'year': int(y.group(1)) if y else None, 'text': text,
                 'note': 'already in the T2 human pool (human-source-mix-v1); dedupe before use'}
            if never_train_hit(w):
                stats['never_train'] += 1; continue
            wins.append(w)
    stats['windows'] = len(wins)
    return wins, stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('parquet'); ap.add_argument('hsm'); ap.add_argument('exclude'); ap.add_argument('--overlap')
    a = ap.parse_args(); rng = random.Random(20261012)
    USED = used_paragraph_hashes()
    arch, s1 = archive_windows(a.parquet, a.exclude)
    pw, s3 = paired_windows(a.exclude, USED); arch = arch + pw
    hsm, s2 = hsm_windows(a.hsm, a.exclude)
    if not a.overlap:  # pass 1: snippets for the H200 check against training / eval windows
        sn = {}
        for i, w in enumerate(arch + hsm):
            n = re.sub(r'\W+', '', w['text'].lower()); sn[str(i)] = [n[j:j + 60] for j in range(0, max(1, len(n) - 59), 300) if len(n[j:j + 60]) == 60]
        (SCR / 'human-pool-snippets.json').write_text(json.dumps(sn))
        json.dump({'n_paper': len(arch)}, open(SCR / 'human-pool-n.json', 'w'))
        print(json.dumps({'archive': dict(s1), 'paired': dict(s3), 'hsm': dict(s2)}, indent=1)); sys.exit()
    hits = json.loads(Path(a.overlap).read_text())
    bad_train, bad_eval = set(hits['training']), set(hits['eval'])
    allw = arch + hsm; keep_a, keep_h = [], []
    for i, w in enumerate(allw):
        if str(i) in bad_eval or str(i) in bad_train:
            continue
        (keep_a if i < len(arch) else keep_h).append(w)
    rng.shuffle(keep_a); rng.shuffle(keep_h)
    for k, w in enumerate(keep_a + keep_h):
        w['id'] = f"human-pool-{k + 1:05d}"; w['dataset'] = 'human_paper_text'; w['label'] = 0
        w['regions'] = [{'start': 0, 'end': len(w['text']), 'label': 0}]; w['group'] = 'paper:' + w['paper_id']
    with gzip.open(HERE / 'human-paper-windows-v1.jsonl.gz', 'wt') as f:
        for w in keep_a:
            f.write(json.dumps(w, ensure_ascii=False) + '\n')
    with gzip.open(HERE / 'human-hsm-windows-v1.jsonl.gz', 'wt') as f:
        for w in keep_h:
            f.write(json.dumps(w, ensure_ascii=False) + '\n')
    st = {'target_windows': TARGET, 'archive_windows': len(keep_a), 'archive_papers': len({w['paper_id'] for w in keep_a}),
          'paper_windows_by_source': dict(Counter(w['source'] for w in keep_a)),
          'archive_by_venue': dict(Counter(f"{w['conference']} {w['year']}" for w in keep_a)),
          'hsm_windows_flagged': len(keep_h), 'hsm_by_source': dict(Counter(w['source'] for w in keep_h)),
          'dropped_overlap_training': sum(1 for i in range(len(allw)) if str(i) in bad_train), 'dropped_overlap_eval': sum(1 for i in range(len(allw)) if str(i) in bad_eval),
          'chars_median_archive': sorted(len(w['text']) for w in keep_a)[len(keep_a) // 2] if keep_a else None,
          'max_chars': MAX_CHARS, 'token_note': '~4 chars/token -> <=~475 tokens; prep crops at 510', 'archive_prep': dict(s1), 'paired_prep': dict(s3), 'hsm_prep': dict(s2)}
    (HERE / 'human-pool-stats.json').write_text(json.dumps(st, indent=1)); print(json.dumps(st, indent=1))
