"""LLM-written one- and two-sentence edits inside human paper paragraphs, with exact AI character spans.

Source: papers dated 2022 or earlier from open-text-detector/paper-text (user rule: human-written),
excluding NeurIPS 2021 (the only venue/year shared with the paired training and evaluation papers) and any
paragraph whose text also appears in the training, selection, calibration or evaluation windows.
Paragraphs are rebuilt from the Poppler text the same way as the paired data: joined lines, line-break
hyphens removed, whitespace normalized, paragraphs separated by blank lines.

Each candidate gets one edit type (rewrite one sentence, rewrite two adjacent sentences, insert a sentence,
merge two sentences or split one). Luna returns the full edited paragraph; sentences are aligned against the
original and the edit is kept only if exactly one contiguous block changed, of the requested size. Only the
changed block's characters are labeled AI.

Usage:
  build_llm_edits.py prepare ARCHIVE_PARQUET OUT_DIR           # candidates.jsonl (+ snippets for overlap check)
  build_llm_edits.py generate OUT_DIR --target 200 --max-calls 320 --budget 5
Calls go to OpenRouter's OpenAI Flex endpoint only (no fallback); the key is read from
benchmarks/otd/.env.secrets into process memory and never written or printed.
"""
import argparse, asyncio, difflib, gzip, json, os, random, re, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder of the Oct 6 session that made the intermediate files (outside the repo); set SCRATCH to it

import requests

MODEL = 'openai/gpt-6-luna'
PROVIDER = {'only': ['openai/flex'], 'allow_fallbacks': False, 'require_parameters': True,
            'max_price': {'prompt': .05, 'completion': .25, 'request': 0}}
SETTINGS = {'max_tokens': 4000, 'reasoning': {'effort': 'low', 'exclude': True}, 'response_format': {'type': 'json_object'}}
EDITS = ['rewrite_one', 'rewrite_two', 'insert_one', 'merge_two', 'split_one']
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
WORD = re.compile(r'[a-z0-9]+')
ROOT = Path(__file__).resolve().parents[2]


# ---------- paragraph reconstruction from Poppler text ----------

def paragraphs(raw):
    """Join Poppler lines into paragraphs: a short line ending a sentence, a blank line or a page break ends one."""
    lines = [l.rstrip() for l in raw.replace('\x0c', '\n\x0c\n').split('\n')]
    body = [len(l) for l in lines if len(l.strip()) > 40]
    full = statistics.median(body) if body else 80
    out, cur = [], []

    def flush():
        if cur:
            t = ''
            for l in cur:
                l = l.strip()
                if t.endswith('-') and l[:1].islower():
                    t = t[:-1] + l  # line-break hyphen
                else:
                    t = (t + ' ' + l) if t else l
            out.append(re.sub(r'\s+', ' ', t).strip())
            cur.clear()

    for l in lines:
        s = l.strip()
        if re.fullmatch(r'(\d+\s*)?(References|REFERENCES|Bibliography|Acknowledge?ments?|ACKNOWLEDGE?MENTS?)', s):
            break  # stop at the reference list / acknowledgements
        if not s or s == '\x0c':
            flush(); continue
        if len(l) < 0.6 * full and not re.search(r'[.!?:,;]["”)]?$', s) and len(s.split()) <= 10:
            flush(); out.append(s); continue  # short unpunctuated line: section heading, kept as its own fragment
        if re.fullmatch(r'\d{1,3}', s) or s.startswith(('Published in Transactions', 'Published as a conference paper', 'Under review as')):
            continue  # page numbers and running headers
        cur.append(s)
        if re.search(r'[.!?:]["”)]?$', s) and len(l) < 0.8 * full:
            flush()
    flush()
    return out


def prose(p, lo=350, hi=1300):
    if '\ufffd' in p:
        return False  # undecodable glyphs from the PDF
    if not (lo <= len(p) <= hi) or not p[:1].isupper() or not re.search(r'[.!?]["”)]?$', p):
        return False
    letters = sum(c.isalpha() for c in p) / len(p)
    singles = sum(len(w) == 1 for w in p.split()) / max(1, len(p.split()))
    if len(re.findall(r'\b(19|20)\d\d[a-z]?\b', p)) >= 4 or 'arXiv:' in p:
        return False  # citation lists
    return letters > .76 and singles < .12 and len(sentences(p)) >= 3 and not re.search(r'(Figure|Table|Fig\.) \d+[:.]', p[:20])


def sentences(text):
    return [(m.start(), m.end()) for m in SENT.finditer(text) if m.group().strip()]


def words(s):
    return WORD.findall(s.lower())


def sim(a, b):
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.
    u = len(set(wa) & set(wb)) / len(set(wa) | set(wb))
    ba, bb = set(zip(wa, wa[1:])), set(zip(wb, wb[1:]))
    return (u + (len(ba & bb) / len(ba | bb) if ba | bb else 0.)) / 2


def prepare(parquet, out):
    import pyarrow.parquet as pq
    rng = random.Random(20261005)
    t = pq.read_table(parquet, columns=['paper_id', 'title', 'conference', 'year', 'text', 'extraction_method'])
    papers = [r for r in t.to_pylist() if r['year'] <= 2022]
    cands = []
    for r in papers:
        ps = paragraphs(r['text'])
        idx = [i for i in range(1, len(ps) - 1) if prose(ps[i]) and len(ps[i - 1]) >= 120 and len(ps[i + 1]) >= 120]
        rng.shuffle(idx)
        for i in idx[:2]:  # at most two paragraphs per paper
            before, target, after = ps[i - 1][-900:], ps[i], ps[i + 1][:900]
            before = before[before.find(' ') + 1:] if len(ps[i - 1]) > 900 else before
            after = after[:after.rfind(' ')] if len(ps[i + 1]) > 900 else after
            cands.append({'cand_id': f"{r['paper_id'][:16]}/p{i:03d}", 'paper_id': r['paper_id'], 'title': r['title'],
                          'conference': r['conference'], 'year': r['year'], 'extraction_method': r['extraction_method'],
                          'before': before, 'target': target, 'after': after})
    rng.shuffle(cands)
    for k, c in enumerate(cands):
        c['edit_type'] = EDITS[k % len(EDITS)]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'candidates.jsonl', 'w') as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(len(papers), 'papers <=2022 (NeurIPS 2021 excluded);', len(cands), 'candidate paragraphs from',
          len({c['paper_id'] for c in cands}), 'papers;', Counter(c['conference'] + str(c['year']) for c in cands))


V1_MIX = ['rewrite_one', 'rewrite_two', 'insert_one'] * 3 + ['split_one']  # 30/30/30/10; merge dropped after the pilot
PAIRED = ROOT / 'research/data/paper-gap10000-v3-luna-20260930'
SUITE = ROOT / 'research/evaluation/space-suite-v1/package'


def trim(before, after):
    b_ = before[-900:]; b_ = b_[b_.find(' ') + 1:] if len(before) > 900 else b_
    a_ = after[:900]; a_ = a_[:a_.rfind(' ')] if len(after) > 900 else a_
    return b_, a_


def snip(t):
    n = re.sub(r'\W+', '', t.lower())
    return [x for x in (n[:60], n[len(n) // 2:len(n) // 2 + 60]) if len(x) == 60]


def prepare_v1(parquet, out, excluded_ids_json):
    """Archive papers <=2022 (no per-paper cap) + non-target paragraphs of the paired-data TRAIN papers.

    excluded_ids_json: {"paper_ids": [...], "text_overlap_cand_ids": [...]} from the H200 overlap check against
    selection, calibration and sweep-evaluation windows (training-window overlap is reported, not excluded, for
    paired-paper context paragraphs). The frozen space-suite-v1 package is checked here by paper id and text."""
    import sys
    import pyarrow.parquet as pq
    rng = random.Random(20261006)
    cands, stats = [], Counter()
    # 1. archive papers dated <=2022 (NeurIPS 2021 excluded), every eligible paragraph
    t = pq.read_table(parquet, columns=['paper_id', 'forum_id', 'title', 'conference', 'year', 'text', 'extraction_method'])
    for r in t.to_pylist():
        if r['year'] > 2022:
            continue
        ps = paragraphs(r['text'])
        for i in range(1, len(ps) - 1):
            if prose(ps[i]) and len(ps[i - 1]) >= 120 and len(ps[i + 1]) >= 120:
                b_, a_ = trim(ps[i - 1], ps[i + 1])
                cands.append({'cand_id': f"{r['paper_id'][:16]}/p{i:03d}", 'paper_id': r['paper_id'], 'forum_id': r['forum_id'],
                              'source': 'archive_pre2023', 'title': r['title'], 'conference': r['conference'], 'year': r['year'],
                              'extraction_method': r['extraction_method'], 'before': b_, 'target': ps[i], 'after': a_})
    # 2. paired-data train papers: the same Poppler paragraph extraction and target filter as the paired data, minus its targets
    sys.path.insert(0, str(ROOT / 'benchmarks/otd'))
    import paper_gap50 as g, paper_gap250_sources as srcmod
    g.OUT = PAIRED
    papers = [json.loads(l) for l in open(PAIRED / 'papers.jsonl')]
    held = {}
    for l in open(PAIRED / 'passages.jsonl'):
        x = json.loads(l); held.setdefault(x['paper_id'], set()).add(re.sub(r'\W+', '', x['held_out'].lower()))
    for p_ in papers:
        if p_['split'] != 'train':
            continue
        try:
            ps = srcmod.paragraphs(p_)
        except Exception:
            stats['paired_extract_error'] += 1; continue
        for i in range(1, len(ps) - 1):
            t_ = ps[i]['text']
            if not srcmod.target(ps[i]) or '\ufffd' in t_ or len(sentences(t_)) < 3:
                continue
            if re.sub(r'\W+', '', t_.lower()) in held.get(p_['paper_id'], set()):
                stats['paired_target_paragraph_skipped'] += 1; continue
            a, z = ps[i - 1]['text'], ps[i + 1]['text']
            if len(a.split()) < 20 or len(z.split()) < 20 or z_page_gap(ps[i - 1], ps[i + 1]):
                continue
            b_, a_ = trim(a, z)
            cands.append({'cand_id': f"{p_['paper_id']}/q{i:03d}", 'paper_id': p_['paper_id'], 'forum_id': None,
                          'source': 'paired_train_nontarget', 'title': p_['title'], 'conference': p_['conference'], 'year': p_['year'],
                          'extraction_method': 'poppler-bbox-layout (paired-data extractor)', 'before': b_, 'target': t_, 'after': a_})
    stats['raw_candidates'] = len(cands)
    # 3. exclusions
    ex = json.loads(Path(excluded_ids_json).read_text())
    bad_papers = set(ex['paper_ids']); bad_cands = set(ex['text_overlap_cand_ids'])
    suite_ids, suite_snips = set(), set()
    import gzip as gz
    for f in SUITE.glob('*.jsonl.gz'):
        for l in gz.open(f, 'rt'):
            x = json.loads(l)
            for k in ('paper_id', 'group_id', 'forum_id'):
                if x.get(k) not in (None, 'None', ''):
                    suite_ids.add(str(x[k]).split('/')[0])
            n = re.sub(r'\W+', '', x['text'].lower())
            suite_snips.update(n[i:i + 60] for i in range(0, max(1, len(n) - 59), 30))
    pilot = {json.loads(l)['id'].rsplit('/', 1)[0] for l in gzip.open(out / 'llm-edits-pilot.jsonl.gz', 'rt')}
    keep = []
    for c in cands:
        ids = {c['paper_id'], str(c.get('forum_id'))}
        if ids & bad_papers:
            stats['excluded_paper_in_eval_selection_calibration'] += 1; continue
        if ids & suite_ids:
            stats['excluded_paper_in_space_suite'] += 1; continue
        if c['cand_id'] in bad_cands:
            stats['excluded_text_overlap_eval_selection_calibration'] += 1; continue
        n = re.sub(r'\W+', '', c['target'].lower())
        if any(n[i:i + 60] in suite_snips for i in range(0, max(1, len(n) - 59), 15)):
            stats['excluded_text_overlap_space_suite'] += 1; continue
        if c['cand_id'] in pilot:
            stats['excluded_used_in_pilot'] += 1; continue
        keep.append(c)
    rng.shuffle(keep)
    for k, c in enumerate(keep):
        c['edit_type'] = V1_MIX[k % len(V1_MIX)]
    with open(out / 'candidates-v1.jsonl', 'w') as f:
        for c in keep:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    stats['kept'] = len(keep)
    stats.update({f'kept_{k}': v for k, v in Counter(c['source'] for c in keep).items()})
    (out / 'prepare-v1-stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def z_page_gap(a, z):
    return z.get('page', 0) - a.get('page', 0) > 2


# ---------- held-out multi-writer evaluation set ----------

WORKFLOW = ROOT / 'research/data/paper-eval-workflows-luna-20260930'
EVAL_PAPER_DATASETS = {'paper_v3_target', 'paper_workflow_reconstruction', 'human_paper_workflow_matched', 'human_paper_workflow_remaining'}
HELDOUT_QUOTA = {'luna': 400, 'haiku': 350, 'sonnet': 350, 'opus': 350}
NEVER_TRAIN = Path(__file__).resolve().parent / 'heldout-eval-never-train.json'
# Every held-out list; never_train_hit() checks all that exist (same format: paper_ids, paragraph_sha256, hashed shingles).
NEVER_TRAIN_FILES = [NEVER_TRAIN, ROOT / 'research/claude-fullpapers-20261006/heldout-never-train.json',
                     ROOT / 'research/claude-sections-20261006/heldout-never-train.json',
                     ROOT / 'research/claude-hosted-edits-20261006/heldout-never-train.json',
                     ROOT / 'research/claude-fable-evals-20261006/heldout-never-train.json',
                     # T2.1 held-out split (Oct 6 evening): GPT-6 Sol and Fable papers plus 15% of Opus/Sonnet papers.
                     ROOT / 'research/t21-heldout-20261006/t21-heldout-never-train.json',
                     ROOT / 'research/t21-heldout-20261006/t21-heldout-never-train-v2.json',
                     # wide1 (Oct 7): GLM/MiMo eval-only papers and the 15% DeepSeek held-out papers.
                     ROOT / 'research/corpus-sampling-20261007/wide1-heldout-never-train.json']


def norm_hash(t):
    import hashlib
    return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()


def never_train_hit(row, _cache={}):
    """For training builders: True if a row must not be used for training (held-out eval paper or paragraph).
    Loads every file in NEVER_TRAIN_FILES (sentence-edit, full-paper and section held-out lists). Also checks row['id'] parts.

    Checks the row's paper_id/group (with or without a 'paper:' prefix) against the never-train paper list, and every
    blank-line-separated paragraph of row['text'] (normalized: lowercase, non-word characters removed) against the
    paragraph hashes. Rows whose paragraphs were re-split differently should additionally be screened with the
    hashed 60-character normalized shingles ('paragraph_shingle_sha256_12', stored at every 10th offset; the row is queried
    at every offset, so any copied held-out stretch of 70+ normalized characters is found)."""
    if not _cache:
        _cache.update(papers=set(), hashes=set(), shingles=set())
        for f in NEVER_TRAIN_FILES:
            if f.exists():
                d = json.loads(f.read_text())
                _cache['papers'] |= set(d['paper_ids']); _cache['hashes'] |= set(d['paragraph_sha256'])
                _cache['shingles'] |= set(d['paragraph_shingle_sha256_12'])
    ids = {str(row.get(k, '')).replace('paper:', '').split('/')[0] for k in ('paper_id', 'group', 'group_id', 'seed_id', 'item_id')}
    ids |= {x for x in re.split(r'[/]', str(row.get('id', ''))) if x.startswith(('claude-fp-', 'claude-sec-'))} | {str(row.get('id', '')).replace('fullpaper-', '')}
    if ids & _cache['papers']:
        return True
    text = row.get('text', '')
    # Skip short blocks (headings like '## 1 Introduction', boilerplate lines): they recur across unrelated documents.
    if any(norm_hash(p) in _cache['hashes'] for p in text.split('\n\n') if len(re.sub(r'\W+', '', p)) >= 100):
        return True
    import hashlib
    n = re.sub(r'\W+', '', text.lower())
    return any(hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] in _cache['shingles'] for i in range(max(0, len(n) - 59)))


def prepare_heldout(parquet, out, eval_ids_json, overlap_json=None):
    """Raw pass (overlap_json None): every eligible paragraph of the sweep-eval papers and of unused <=2022 archive
    papers -> candidates-heldout-raw.jsonl (+ snippets for the H200 check). Final pass: drop overlaps, pick one paragraph
    per paper per round (round-robin, maximal paper spread), assign writers and edit types, write never-train list."""
    import sys
    raw_path = out / 'candidates-heldout-raw.jsonl'
    if overlap_json is None:
        import pyarrow.parquet as pq
        ev = json.loads(Path(eval_ids_json).read_text())  # {"paper_ids": [...]} from sweep-eval-rows paper datasets
        evp = set(ev['paper_ids'])
        used = set()
        for f in ('llm-edits-pilot.jsonl.gz', 'llm-edits-v1.jsonl.gz'):
            used |= {json.loads(l)['paper_id'] for l in gzip.open(out / f, 'rt')}
        sys.path.insert(0, str(ROOT / 'benchmarks/otd'))
        import paper_gap50 as g, paper_gap250_sources as srcmod
        cands, stats = [], Counter()
        targets = set()
        for d in (PAIRED, WORKFLOW):
            for l in open(d / 'passages.jsonl'):
                x = json.loads(l)
                for k in ('held_out', 'target', 'original_target'):
                    if isinstance(x.get(k), str):
                        targets.add(norm_hash(x[k]))
        for d in (PAIRED, WORKFLOW):
            g.OUT = d
            for p_ in map(json.loads, open(d / 'papers.jsonl')):
                if p_['paper_id'] not in evp or not (d / 'sources' / f"{p_['paper_id']}.bbox.html").exists():
                    continue
                stats['eval_papers_with_sources'] += 1
                try:
                    ps = srcmod.paragraphs(p_)
                except Exception:
                    stats['extract_error'] += 1; continue
                for i in range(1, len(ps) - 1):
                    t_ = ps[i]['text']
                    if not srcmod.target(ps[i]) or '\ufffd' in t_ or len(sentences(t_)) < 3:
                        continue
                    if norm_hash(t_) in targets:
                        stats['skipped_existing_target_paragraph'] += 1; continue
                    a, z = ps[i - 1]['text'], ps[i + 1]['text']
                    if len(a.split()) < 20 or len(z.split()) < 20 or z_page_gap(ps[i - 1], ps[i + 1]):
                        continue
                    b_, a_ = trim(a, z)
                    cands.append({'cand_id': f"{p_['paper_id']}/h{i:03d}", 'paper_id': p_['paper_id'], 'source': 'sweep_eval_paper',
                                  'conference': p_['conference'], 'year': p_['year'], 'before': b_, 'target': t_, 'after': a_})
        t = pq.read_table(parquet, columns=['paper_id', 'forum_id', 'conference', 'year', 'text'])
        for r in t.to_pylist():
            if r['year'] > 2022 or r['paper_id'] in used:
                continue
            stats['unused_archive_papers'] += 1
            ps = paragraphs(r['text'])
            for i in range(1, len(ps) - 1):
                if prose(ps[i]) and len(ps[i - 1]) >= 120 and len(ps[i + 1]) >= 120:
                    b_, a_ = trim(ps[i - 1], ps[i + 1])
                    cands.append({'cand_id': f"{r['paper_id'][:16]}/h{i:03d}", 'paper_id': r['paper_id'], 'forum_id': r['forum_id'],
                                  'source': 'archive_pre2023_unused', 'conference': r['conference'], 'year': r['year'],
                                  'before': b_, 'target': ps[i], 'after': a_})
        with open(raw_path, 'w') as f:
            for c in cands:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        (Path(os.environ.get('HELDOUT_SNIPPETS', '/tmp/heldout-snippets.json'))).write_text(json.dumps({c['cand_id']: snip(c['target']) for c in cands}))
        stats['raw_candidates'] = len(cands); stats.update({f'raw_{k}': v for k, v in Counter(c['source'] for c in cands).items()})
        stats['raw_papers'] = len({c['paper_id'] for c in cands})
        (out / 'heldout-prepare-stats.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))
        return
    # final pass
    rng = random.Random(20261007)
    cands = [json.loads(l) for l in open(raw_path)]
    ov = json.loads(Path(overlap_json).read_text())
    drop = set(ov['training']) | set(ov['selection_calibration']) | set(ov['sweep_eval_rows'])
    stats = json.loads((out / 'heldout-prepare-stats.json').read_text())
    stats.update({'overlap_training_windows': len(ov['training']), 'overlap_selection_calibration': len(ov['selection_calibration']),
                  'overlap_sweep_eval_rows': len(ov['sweep_eval_rows']), 'overlap_space_suite_kept': len(ov.get('space_suite', []))})
    seen_h, by_paper = set(), {}
    for c in cands:
        h = norm_hash(c['target'])
        if c['cand_id'] in drop or h in seen_h:
            continue
        seen_h.add(h); by_paper.setdefault(c['paper_id'], []).append(c)
    papers = sorted(by_paper); rng.shuffle(papers)
    for v in by_paper.values():
        rng.shuffle(v)
    need = sum(HELDOUT_QUOTA.values()); chosen = []; rnd = 0
    while len(chosen) < need and any(len(v) > rnd for v in by_paper.values()):
        for pid in papers:
            if len(by_paper[pid]) > rnd and len(chosen) < need:
                chosen.append(by_paper[pid][rnd])
        rnd += 1
    if len(chosen) < need:
        raise SystemExit(f'only {len(chosen)} eligible held-out paragraphs for {need}')
    left = dict(HELDOUT_QUOTA); order = [w for w in ('luna', 'haiku', 'sonnet', 'opus')]; k = 0; counts = Counter()
    for c in chosen:
        while not left[order[k % 4]]:
            k += 1
        w = order[k % 4]; left[w] -= 1; k += 1
        c['writer'] = w; c['edit_type'] = V1_MIX[counts[w] % len(V1_MIX)]; counts[w] += 1
    with open(out / 'candidates-heldout.jsonl', 'w') as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    # never-train list: every held-out eval paper (whether or not a paragraph was chosen) and all chosen paragraphs
    all_papers = sorted({c['paper_id'] for c in cands} | set(json.loads(Path(eval_ids_json).read_text())['paper_ids']))
    import hashlib
    sh = sorted({hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] for c in chosen for n in [re.sub(r'\W+', '', c['target'].lower())]
                 for i in range(0, len(n) - 59, 10)})  # 60-char shingles every 10th offset, hashed: no paper text is stored
    NEVER_TRAIN.write_text(json.dumps({
        'purpose': 'Held-out multi-writer small-edit evaluation (heldout-eval-v1). Never train on these papers or paragraphs.',
        'created_pdt': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
        'check': 'build_llm_edits.never_train_hit(row): paper_id/group match, normalized paragraph sha256 per blank-line paragraph, then hashed 60-char normalized shingles (sha256 hex[:12], stored every 10th offset, queried at every offset).',
        'normalization': "re.sub(r'\\W+', '', text.lower())", 'paper_ids': all_papers,
        'paragraph_sha256': sorted({norm_hash(c['target']) for c in chosen}), 'paragraph_shingle_sha256_12': sh}, separators=(',', ':')))
    stats.update({'chosen': len(chosen), 'chosen_papers': len({c['paper_id'] for c in chosen}), 'rounds': rnd,
                  'chosen_by_source': dict(Counter(c['source'] for c in chosen)),
                  'chosen_by_writer_type': {w: dict(Counter(c['edit_type'] for c in chosen if c['writer'] == w)) for w in HELDOUT_QUOTA},
                  'never_train_papers': len(all_papers)})
    (out / 'heldout-prepare-stats.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))


def write_claude_batches(out, dest, per_file=50):
    dest.mkdir(parents=True, exist_ok=True)
    cands = [json.loads(l) for l in open(out / 'candidates-heldout.jsonl')]
    for w in ('haiku', 'sonnet', 'opus'):
        xs = [c for c in cands if c['writer'] == w]
        for b in range(0, len(xs), per_file):
            with open(dest / f'{w}-{b // per_file + 1:02d}.jsonl', 'w') as f:
                for c in xs[b:b + per_file]:
                    sents = [f'[{i + 1}] ' + c['target'][x:y] for i, (x, y) in enumerate(sentences(c['target']))]
                    f.write(json.dumps({'id': c['cand_id'], 'edit_type': c['edit_type'], 'instruction': claude_instruction(c),
                                        'prev_context': c['before'], 'paragraph': c['target'], 'numbered_sentences': sents,
                                        'next_context': c['after']}, ensure_ascii=False) + '\n')
    print('wrote batches to', dest)


def claude_instruction(c):
    rng = random.Random(c['cand_id']); a, b = plan(c, rng)
    c['_plan'] = [a, b]
    return INSTRUCTIONS[c['edit_type']].format(a=a + 1, b=b + 1)


CLAUDE_TRAIN_QUOTA = {'sonnet': 2500, 'opus': 2500, 'haiku': 1000}


def mathy(t):
    """Mostly-math paragraphs: many math symbols/Greek letters or many single-character tokens (variables)."""
    import unicodedata
    sym = sum(unicodedata.category(ch) in {'Sm', 'So'} or 'GREEK' in unicodedata.name(ch, '') for ch in t)
    toks = t.split(); singles = sum(len(w.strip('.,;:()')) == 1 and w.strip('.,;:()') not in ('a', 'A', 'I') for w in toks)
    return sym / max(1, len(t)) > .015 or singles / max(1, len(toks)) > .08


def reference_like(t):
    return bool(re.search(r'Proceedings of the|In Advances in|arXiv preprint|In International Conference|pp\. \d|Journal of [A-Z]', t)) or \
        len(re.findall(r'\b(19|20)\d\d[a-z]?\b', t)) >= 6


def prepare_claude_train(out, dest, per_file=50):
    """Unused v1 candidates (never accepted in pilot/v1) -> Claude writer batches for training edits."""
    rng = random.Random(20261006)
    accepted = set()
    for f in ('llm-edits-pilot.jsonl.gz', 'llm-edits-v1.jsonl.gz'):
        accepted |= {json.loads(l)['id'].rsplit('/', 1)[0] for l in gzip.open(out / f, 'rt')}
    stats = Counter(); keep = []  # counts; converted to dict before adding details
    for c in map(json.loads, open(out / 'candidates-v1.jsonl')):
        if c['cand_id'] in accepted:
            stats['already_accepted_pilot_or_v1'] += 1; continue
        stats['unused'] += 1
        if never_train_hit({'paper_id': c['paper_id'], 'text': c['target']}) or never_train_hit({'text': c['before'] + '\n\n' + c['after']}):
            stats['excluded_never_train'] += 1; continue
        if c['year'] > 2022:
            stats['excluded_after_2022'] += 1; continue
        if reference_like(c['target']):
            stats['excluded_reference_like'] += 1; continue
        if mathy(c['target']):
            stats['excluded_mostly_math'] += 1; continue
        keep.append(c)
    rng.shuffle(keep)
    total = sum(CLAUDE_TRAIN_QUOTA.values()); n = min(total, len(keep))
    quota = {w: int(q * n / total) for w, q in CLAUDE_TRAIN_QUOTA.items()}
    stats['eligible'] = len(keep); stats['scaled_down'] = n < total
    chosen, k = [], 0
    for w in ('sonnet', 'opus', 'haiku'):
        for i, c in enumerate(keep[k:k + quota[w]]):
            c = dict(c, writer=w, edit_type=V1_MIX[i % len(V1_MIX)]); chosen.append(c)
        k += quota[w]
    with open(out / 'candidates-claude-v1.jsonl', 'w') as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    dest.mkdir(parents=True, exist_ok=True)
    for w in ('sonnet', 'opus', 'haiku'):
        xs = [c for c in chosen if c['writer'] == w]
        for b in range(0, len(xs), per_file):
            with open(dest / f'{w}-{b // per_file + 1:03d}.jsonl', 'w') as f:
                for c in xs[b:b + per_file]:
                    sents = [f'[{i + 1}] ' + c['target'][x:y] for i, (x, y) in enumerate(sentences(c['target']))]
                    f.write(json.dumps({'id': c['cand_id'], 'edit_type': c['edit_type'], 'instruction': claude_instruction(c),
                                        'prev_context': c['before'], 'paragraph': c['target'], 'numbered_sentences': sents,
                                        'next_context': c['after']}, ensure_ascii=False) + '\n')
    stats = dict(stats); stats.update({'chosen': len(chosen), 'quota': quota, 'papers': len({c['paper_id'] for c in chosen}),
                  'by_source': dict(Counter(c['source'] for c in chosen)),
                  'by_writer_type': {w: dict(Counter(c['edit_type'] for c in chosen if c['writer'] == w)) for w in quota},
                  'batch_files': {w: -(-quota[w] // per_file) for w in quota}})
    (out / 'claude-train-prepare-stats.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))


CLAUDE_GEN = {'haiku': 'claude-haiku-4-5', 'sonnet': 'claude-sonnet-5-5', 'opus': 'claude-opus-5-5', 'fable': 'claude-fable-5-1'}


def ingest_claude(out, writer, indir, split='heldout'):
    """Align Claude writer outputs ({"id", "edited_paragraph"} JSONL files) with the originals; append accepted rows.
    split='heldout' -> heldout-eval-v1.jsonl.gz (evaluation); split='train' -> llm-edits-claude-v1.jsonl.gz (training)."""
    cfile = 'candidates-heldout.jsonl' if split == 'heldout' else 'candidates-claude-v1.jsonl'
    cands = {c['cand_id']: c for c in map(json.loads, open(out / cfile)) if c['writer'] == writer}
    got, rejects = {}, Counter()
    for f in sorted(Path(indir).glob('*.jsonl')):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                x = json.loads(l)
            except ValueError:
                rejects['bad_json_line'] += 1; continue
            if x.get('id') in cands and x['id'] not in got:
                got[x['id']] = x.get('edited_paragraph', '')
    rows = []
    for cid, edited in got.items():
        c = cands[cid]; rng = random.Random(cid); a, b = plan(c, rng)
        res, reason = strict_edit(c['target'], edited, c['edit_type'], a, b)
        if not res:
            rejects[reason] += 1; continue
        row = make_row(c, a, b, *res, {'service_tier': None, 'cost_usd': None})
        row['generator'] = CLAUDE_GEN[writer]
        if split == 'heldout':
            row['dataset'] = 'papers_llm_edit_heldout_eval'; row['eval_source'] = c['source']
        else:
            row['source'] = c['source']
            if never_train_hit(row):
                rejects['never_train_hit'] += 1; continue
        row['id'] = f"{cid}/{writer}-{c['edit_type']}"
        rows.append(row)
    path = out / ('heldout-eval-v1.jsonl.gz' if split == 'heldout' else 'llm-edits-claude-v1.jsonl.gz')
    old = [json.loads(l) for l in gzip.open(path, 'rt')] if path.exists() else []
    old = [r for r in old if r['generator'] != CLAUDE_GEN[writer]]
    with gzip.open(path, 'wt') as f:
        for r in old + rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rep_ = {'writer': CLAUDE_GEN[writer], 'assigned': len(cands), 'returned': len(got), 'accepted': len(rows),
            'acceptance_of_returned': round(len(rows) / max(1, len(got)), 3), 'acceptance_of_assigned': round(len(rows) / max(1, len(cands)), 3),
            'rejects': dict(rejects), 'by_edit_type': dict(Counter(r['edit_type'] for r in rows)),
            'ai_span_median': statistics.median([len(r['inserted_ai']) for r in rows]) if rows else None}
    m = json.loads((out / 'manifest.json').read_text())
    m.setdefault('heldout_eval_v1' if split == 'heldout' else 'claude_train_v1', {})[CLAUDE_GEN[writer]] = rep_
    (out / 'manifest.json').write_text(json.dumps(m, indent=1)); print(json.dumps(rep_, indent=1))


# ---------- generation ----------

SYSTEM = ('You are helping build a research dataset on AI-assisted editing of scientific papers. You edit a paragraph from a '
          'real paper the way an author using an AI writing assistant would. Keep the paper\'s voice, terminology, notation '
          'and citation style. Change only what the instruction asks; copy every other sentence exactly, character for character.')
INSTRUCTIONS = {
    'rewrite_one': 'Rewrite sentence {a} so it reads more polished and fluent, keeping its meaning. The new version should be noticeably reworded, not a one-word change.',
    'rewrite_two': 'Rewrite sentences {a} and {b} (adjacent) so they read more polished and fluent, keeping their meaning. Reword them noticeably; you may restructure them, but still produce two sentences.',
    'insert_one': 'Insert one new sentence directly after sentence {a} that adds a natural, plausible clarification or transition consistent with the paragraph. Do not change any existing sentence.',
    'merge_two': 'Merge sentences {a} and {b} (adjacent) into a single well-written sentence that keeps their meaning.',
    'split_one': 'Split sentence {a} into two well-written sentences that keep its meaning, rewording as needed.',
}
EXPECT = {'rewrite_one': (1, {1}), 'rewrite_two': (2, {2}), 'insert_one': (0, {1}), 'merge_two': (2, {1}), 'split_one': (1, {2})}


def plan(c, rng):
    S = sentences(c['target']); n = len(S)
    if c['edit_type'] in ('rewrite_two', 'merge_two'):
        a = rng.randrange(n - 1); return a, a + 1
    a = rng.randrange(n); return a, a


def prompt(c, a, b):
    S = sentences(c['target'])
    numbered = '\n'.join(f'[{i + 1}] {c["target"][x:y]}' for i, (x, y) in enumerate(S))
    instr = INSTRUCTIONS[c['edit_type']].format(a=a + 1, b=b + 1)
    return (f'{instr}\n\nThe paragraph, split into numbered sentences for reference:\n{numbered}\n\n'
            'Return JSON: {"edited_paragraph": "<the full paragraph after the edit, as plain running text without sentence numbers>"}')


def align_edit(orig, edited, etype):
    """One contiguous changed block of the expected size -> (o0, o1, new_block) in original char offsets, else None."""
    S, E = sentences(orig), sentences(edited)
    so = [orig[x:y] for x, y in S]; se = [edited[x:y] for x, y in E]
    norm = lambda s: re.sub(r'\s+', ' ', s).strip()
    ops = [op for op in difflib.SequenceMatcher(a=[norm(s) for s in so], b=[norm(s) for s in se], autojunk=False).get_opcodes() if op[0] != 'equal']
    if len(ops) != 1:
        return None, f'{len(ops)} changed blocks'
    _, i0, i1, j0, j1 = ops[0]
    n_old, n_new_ok = EXPECT[etype]
    if i1 - i0 != n_old or (j1 - j0) not in n_new_ok:
        return None, f'changed {i1 - i0}->{j1 - j0} sentences'
    new = ' '.join(se[j0:j1]).strip()
    if etype != 'insert_one':
        old = ' '.join(so[i0:i1])
        if sim(old, new) > .9:
            return None, 'too close to original'
        o0, o1 = S[i0][0], S[i1 - 1][1]
    else:
        o0 = o1 = S[i0 - 1][1] if i0 > 0 else 0
    return (o0, o1, new), None


EXPECTED_SENTENCES = {'rewrite_one': 1, 'rewrite_two': 2, 'insert_one': 1, 'split_one': 2}


def named_span(orig, etype, a, b):
    """Character span of the sentence(s) named in the instruction (an insertion point for insert_one)."""
    S = sentences(orig)
    if etype == 'insert_one':
        return S[a][1], S[a][1]
    return S[a][0], S[b if etype == 'rewrite_two' else a][1]


def strict_edit(orig, edited, etype, a, b):
    """Strict check used by every Claude/OpenAI sentence-edit ingest and by selfcheck.py.

    The edit must land on the NAMED sentence(s) (a, b from plan(), same numbering as numbered_sentences; for insert_one
    the new sentence follows sentence a). Everything before and after that span must be byte-identical to the
    original; only the whitespace directly adjacent to the edit may differ. The edited span must have the expected
    sentence count and (for rewrites/splits) be noticeably reworded. Returns ((o0, o1, new), None) or (None, reason)."""
    if etype not in EXPECTED_SENTENCES:
        return None, f'unsupported edit type {etype}'
    o0, o1 = named_span(orig, etype, a, b)
    pre, suf, e = orig[:o0].rstrip(), orig[o1:].lstrip(), (edited or '').strip()
    if not e.startswith(pre):
        return None, 'text before the named sentence changed'
    if len(e) < len(pre) + len(suf) or not e.endswith(suf):
        return None, 'text after the named sentence changed'
    raw = e[len(pre):len(e) - len(suf)]
    if '\n' in raw.strip():
        return None, 'line break inside the edit'
    new = raw.strip()
    if not new:
        return None, 'named sentence deleted' if etype != 'insert_one' else 'nothing inserted'
    n = len(sentences(new))
    if n != EXPECTED_SENTENCES[etype]:
        return None, f'{etype}: edit has {n} sentences, expected {EXPECTED_SENTENCES[etype]}'
    if etype != 'insert_one' and sim(orig[o0:o1], new) > .9:
        return None, 'too close to original'
    return (o0, o1, new), None


def make_row(c, a, b, o0, o1, new, meta):
    t = c['target']
    if o0 == o1:  # insertion
        new_t = t[:o0] + (' ' if o0 else '') + new + ('' if o0 else ' ') + t[o1:]
        s_in = o0 + (1 if o0 else 0)
    else:
        new_t = t[:o0] + new + t[o1:]; s_in = o0
    text = c['before'] + '\n\n' + new_t + '\n\n' + c['after']
    ts = len(c['before']) + 2; s0 = ts + s_in; s1 = s0 + len(new)
    assert text[s0:s1] == new
    size = 'one' if c['edit_type'] in ('rewrite_one', 'insert_one', 'merge_two') else 'two'
    return {'id': f"{c['cand_id']}/llm-{c['edit_type']}", 'paper_id': c['paper_id'], 'group': f"paper:{c['paper_id']}",
            'dataset': 'papers_llm_edit', 'text': text, 'target_start': ts, 'target_end': ts + len(new_t),
            'regions': [{'start': 0, 'end': s0, 'label': 0}, {'start': s0, 'end': s1, 'label': 1}, {'start': s1, 'end': len(text), 'label': 0}],
            'edit_size': size, 'edit_type': c['edit_type'], 'replaced_human': t[o0:o1], 'inserted_ai': new,
            'similarity': round(sim(t[o0:o1], new), 3) if o1 > o0 else None, 'conference': c['conference'], 'year': c['year'],
            'generator': MODEL, **meta}


def call(body, key):
    r = requests.post('https://openrouter.ai/api/v1/chat/completions', json=body, timeout=(20, 600),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {'error': {'type': 'non_json'}}


async def generate(out, target, max_calls, budget, conc=16, tag='pilot', passes=1, pass_wait=600):
    heldout = tag == 'heldout-luna'
    key = next(l.split('=', 1)[1].strip().strip('"\'') for l in open(ROOT / 'benchmarks/otd/.env.secrets') if l.startswith('OPENROUTER_API_KEY='))
    cfile, rfile, kfile = (('candidates.jsonl', 'calls.jsonl', 'llm-edits-pilot.jsonl.gz') if tag == 'pilot'
                           else (f'candidates-{tag}.jsonl', f'calls-{tag}.jsonl', f'llm-edits-{tag}.jsonl.gz'))
    if heldout:
        cfile, rfile, kfile = 'candidates-heldout.jsonl', 'calls-heldout-luna.jsonl', 'heldout-eval-luna.jsonl.gz'
    cands = [json.loads(l) for l in open(out / cfile)]
    if heldout:
        cands = [c for c in cands if c['writer'] == 'luna']
    excl = set(json.loads((out / 'overlap-excluded.json').read_text())) if (out / 'overlap-excluded.json').exists() else set()
    cands = [c for c in cands if c['cand_id'] not in excl]
    rng = random.Random(7)
    accepted, rejects, calls = [], Counter(), []
    spent = 0.; tiers = Counter(); lock = asyncio.Lock(); stop = False
    t_start = time.time()
    prior = [json.loads(l) for l in open(out / rfile)] if tag != 'pilot' and (out / rfile).exists() else []
    if prior:  # resume: earlier calls count toward spend and caps; outage failures are retried in later passes
        accepted = [json.loads(l) for l in gzip.open(out / kfile, 'rt')] if (out / kfile).exists() else []
        spent = sum(float((c.get('usage') or {}).get('cost', 0) or 0) for c in prior); calls = list(prior)
        for c in prior:
            tiers[c['service_tier']] += 1
            if c.get('reject_reason'):
                rejects[c['reject_reason'].split(':')[0] if c['reject_reason'].startswith(('http', 'parse')) else c['reject_reason']] += 1
    outage = lambda c: c.get('reject_reason') and ('Flex processing is temporarily unavailable' in str(c.get('error')) or str(c['reject_reason']).startswith('http'))
    settled = {c['cand_id'] for c in prior if not outage(c)}
    accepted_ids = {r['id'].rsplit('/', 1)[0] for r in accepted}
    queue = [c for c in cands if c['cand_id'] not in settled]
    it = iter(queue)
    live = open(out / rfile, 'a') if tag != 'pilot' else None
    keep_f = gzip.open(out / kfile, 'at') if tag != 'pilot' else None

    async def worker():
        nonlocal spent, stop
        while not stop:
            async with lock:
                if len(accepted) >= target or len(calls) >= max_calls or spent >= budget:
                    stop = True; return
                c = next(it, None)
                if c is None:
                    stop = True; return
                a, b = plan(c, random.Random(c['cand_id']) if heldout else rng)
                calls.append(None); slot = len(calls) - 1
            body = {'model': MODEL, 'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt(c, a, b)}],
                    **SETTINGS, 'service_tier': 'flex', 'provider': PROVIDER}
            t0 = time.time(); retries = 0
            while True:  # transient errors (capacity, empty choices) retry on the same Flex route, never another tier
                status, res = await asyncio.to_thread(call, body, key)
                transient = status in (0, 408, 429, 500, 502, 503, 504) or (status == 200 and not (res.get('choices') if isinstance(res, dict) else None))
                if not transient or retries >= 3:
                    break
                retries += 1; await asyncio.sleep(5 * retries)
            usage = res.get('usage', {}) if isinstance(res, dict) else {}
            rec = {'cand_id': c['cand_id'], 'edit_type': c['edit_type'], 'sentences': [a, b], 'http_status': status,
                   'service_tier': res.get('service_tier', 'not_reported') if isinstance(res, dict) else 'not_reported',
                   'model': res.get('model'), 'generation_id': res.get('id'), 'usage': usage, 'seconds': round(time.time() - t0, 1),
                   'retries': retries, 'error': str(res.get('error'))[:200] if isinstance(res, dict) and res.get('error') else None}
            async with lock:
                spent += float(usage.get('cost', 0) or 0); tiers[rec['service_tier']] += 1
            reason = None
            if status != 200:
                reason = f'http {status}: {str(res.get("error"))[:120]}'
                if status in (401, 402, 403):
                    stop = True
            else:
                try:
                    content = res['choices'][0]['message']['content']; rec['output'] = content
                    edited = re.sub(r'\s+', ' ', json.loads(content)['edited_paragraph']).strip()
                    got, reason = align_edit(c['target'], edited, c['edit_type'])
                    if got and c['cand_id'] not in accepted_ids:
                        row = make_row(c, a, b, *got, {'generation_id': res.get('id'), 'served_model': res.get('model'),
                                                       'service_tier': rec['service_tier'], 'cost_usd': usage.get('cost')})
                        if heldout:
                            row.update(dataset='papers_llm_edit_heldout_eval', eval_source=c['source'], id=f"{c['cand_id']}/luna-{c['edit_type']}")
                        async with lock:
                            if len(accepted) < target:
                                accepted.append(row); accepted_ids.add(c['cand_id'])
                                if keep_f:
                                    keep_f.write(json.dumps(row, ensure_ascii=False) + '\n'); keep_f.flush()
                except Exception as e:
                    reason = f'parse: {type(e).__name__}'
            rec['reject_reason'] = reason
            if 'Flex processing is temporarily unavailable' in str(res.get('error') if isinstance(res, dict) else '') and not reason:
                reason = 'parse: flex_unavailable'
            async with lock:
                calls[slot] = rec
                if live:
                    live.write(json.dumps(rec, ensure_ascii=False) + '\n'); live.flush()
                if reason:
                    rejects[reason.split(':')[0] if reason.startswith(('http', 'parse')) else reason] += 1
                if len(calls) % 25 == 0:
                    print(json.dumps({'calls': len(calls), 'accepted': len(accepted), 'spent_usd': round(spent, 4)}), flush=True)

    for p_ in range(passes):
        await asyncio.gather(*[worker() for _ in range(conc)])
        calls = [c for c in calls if c]
        retry = [c for c in calls if outage(c) and c['cand_id'] not in accepted_ids]
        retry_ids = {c['cand_id'] for c in retry}
        done_ok = {c['cand_id'] for c in calls if not outage(c)}
        retry_ids -= done_ok
        if len(accepted) >= target or spent >= budget or len(calls) >= max_calls or not retry_ids or p_ == passes - 1:
            break
        print(json.dumps({'pass': p_ + 1, 'flex_outage_retry': len(retry_ids), 'wait_s': pass_wait}), flush=True)
        await asyncio.sleep(pass_wait)
        it = iter([c for c in cands if c['cand_id'] in retry_ids]); stop = False
    calls = [c for c in calls if c]
    if tag == 'pilot':
        with open(out / rfile, 'w') as f:
            for c in calls:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        with gzip.open(out / kfile, 'wt') as f:
            for r in accepted:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    else:
        live.close(); keep_f.close()
    spans = [len(r['inserted_ai']) for r in accepted]
    us = [c['usage'] for c in calls]
    man = {'created_pdt': time.strftime('%Y-%m-%d %H:%M', time.localtime()), 'model': MODEL, 'provider': PROVIDER, 'settings': SETTINGS,
           'calls': len(calls), 'accepted': len(accepted), 'acceptance_rate': round(len(accepted) / max(1, len(calls)), 3),
           'reject_reasons': dict(rejects), 'by_edit_type': dict(Counter(r['edit_type'] for r in accepted)),
           'attempts_by_edit_type': dict(Counter(c['edit_type'] for c in calls)),
           'by_edit_size': dict(Counter(r['edit_size'] for r in accepted)), 'papers': len({r['paper_id'] for r in accepted}),
           'ai_span_chars': {'min': min(spans, default=0), 'p25': (statistics.quantiles(spans, n=4)[0] if len(spans) > 3 else None),
                             'median': statistics.median(spans) if spans else None, 'p75': (statistics.quantiles(spans, n=4)[2] if len(spans) > 3 else None),
                             'max': max(spans, default=0), 'under_200': sum(s < 200 for s in spans)},
           'cost_usd_provider_reported': round(sum(float(u.get('cost', 0) or 0) for u in us), 6), 'cost_coverage': sum('cost' in u for u in us),
           'input_tokens': sum(u.get('prompt_tokens', 0) for u in us), 'output_tokens': sum(u.get('completion_tokens', 0) for u in us),
           'served_tiers': dict(tiers), 'wall_seconds': round(time.time() - t_start),
           'source': 'open-text-detector/paper-text batch-001, papers dated <=2022, NeurIPS 2021 excluded, text-overlap checked against training/selection/calibration/eval windows'}
    if tag == 'pilot':
        (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    else:
        old = json.loads((out / 'manifest.json').read_text())
        old = old if 'pilot' in old else {'pilot': old}
        man['by_source'] = dict(Counter(next((c['source'] for c in cands if c['cand_id'] == r['id'].rsplit('/', 1)[0]), '?') for r in accepted))
        man['edit_mix_requested'] = 'rewrite_one/rewrite_two/insert_one 30% each, split_one 10%, merge dropped'
        man['prepare'] = json.loads((out / 'prepare-v1-stats.json').read_text()) if (out / 'prepare-v1-stats.json').exists() else None
        if heldout:
            man.pop('prepare', None); man['by_source'] = dict(Counter(r['eval_source'] for r in accepted))
            old.setdefault('heldout_eval_v1', {})['openai/gpt-6-luna'] = man
            path = out / 'heldout-eval-v1.jsonl.gz'
            prev = [json.loads(l) for l in gzip.open(path, 'rt')] if path.exists() else []
            prev = [r for r in prev if r['generator'] != MODEL]
            with gzip.open(path, 'wt') as f:
                for r in prev + accepted:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        else:
            old[tag] = man
        (out / 'manifest.json').write_text(json.dumps(old, indent=1))
    print(json.dumps(man, indent=1))


FABLE_SCR = (SCRATCH / 'claude-fable')


def prepare_heldout_fable(out, n=210):
    """Held-out Fable writer set: n more paragraphs from held-out candidates not chosen for heldout-eval-v1 (same overlap
    drops), one per paper first, spread over as many papers as possible. Appended to candidates-heldout.jsonl with
    writer 'fable'; paragraph hashes/shingles added to heldout-eval-never-train.json."""
    import hashlib
    rng = random.Random(20261011)
    raw = [json.loads(l) for l in open(out / 'candidates-heldout-raw.jsonl')]
    chosen = [json.loads(l) for l in open(out / 'candidates-heldout.jsonl')]
    if any(c['writer'] == 'fable' for c in chosen):
        raise SystemExit('fable items already prepared')
    hits = json.loads((SCRATCH / 'llmedits/heldout-hits.json').read_text())
    drop = set(hits['training']) | set(hits['selection_calibration']) | set(hits['sweep_eval_rows'])
    seen = {norm_hash(c['target']) for c in chosen}; used_papers = Counter(c['paper_id'] for c in chosen)
    by_paper = defaultdict(list)
    for c in raw:
        h = norm_hash(c['target'])
        if c['cand_id'] in drop or h in seen or reference_like(c['target']) or mathy(c['target']):
            continue
        seen.add(h); by_paper[c['paper_id']].append(c)
    papers = sorted(by_paper, key=lambda p: (used_papers[p], rng.random()))
    for v in by_paper.values():
        rng.shuffle(v)
    picked, rnd = [], 0
    while len(picked) < n and any(len(v) > rnd for v in by_paper.values()):
        for pid in papers:
            if len(by_paper[pid]) > rnd and len(picked) < n:
                picked.append(by_paper[pid][rnd])
        rnd += 1
    for k, c in enumerate(picked):
        c.update(writer='fable', edit_type=V1_MIX[k % len(V1_MIX)])
    with open(out / 'candidates-heldout.jsonl', 'a') as f:
        for c in picked:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    d = json.loads(NEVER_TRAIN.read_text())
    d['paper_ids'] = sorted(set(d['paper_ids']) | {c['paper_id'] for c in picked})
    d['paragraph_sha256'] = sorted(set(d['paragraph_sha256']) | {norm_hash(c['target']) for c in picked})
    d['paragraph_shingle_sha256_12'] = sorted(set(d['paragraph_shingle_sha256_12']) | {hashlib.sha256(n_[i:i + 60].encode()).hexdigest()[:12]
                                              for c in picked for n_ in [re.sub(r'\W+', '', c['target'].lower())] for i in range(0, len(n_) - 59, 10)})
    NEVER_TRAIN.write_text(json.dumps(d, separators=(',', ':')))
    bd = FABLE_SCR / 'heldout'; bd.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(picked), 50):
        with open(bd / f'fable-heldout-{b // 50 + 1:03d}.jsonl', 'w') as f:
            for c in picked[b:b + 50]:
                sents = [f'[{i + 1}] ' + c['target'][x:y] for i, (x, y) in enumerate(sentences(c['target']))]
                f.write(json.dumps({'id': c['cand_id'], 'edit_type': c['edit_type'], 'instruction': claude_instruction(c),
                                    'prev_context': c['before'], 'paragraph': c['target'], 'numbered_sentences': sents,
                                    'next_context': c['after']}, ensure_ascii=False) + '\n')
    print(json.dumps({'items': len(picked), 'papers': len({c['paper_id'] for c in picked}), 'by_source': dict(Counter(c['source'] for c in picked)),
                      'by_type': dict(Counter(c['edit_type'] for c in picked)), 'new_papers_not_in_heldout_v1': sum(1 for c in picked if not used_papers[c['paper_id']])}, indent=1))


def build_score_rows(out):
    """heldout-score-rows.jsonl.gz from heldout-eval-v1 (every writer): one 'mixed' row per edit + one human control per
    distinct original window. Same format as the rows built on Oct 5."""
    cands = {c['cand_id']: c for c in map(json.loads, open(out / 'candidates-heldout.jsonl'))}
    rows, controls = [], {}
    for r in map(json.loads, gzip.open(out / 'heldout-eval-v1.jsonl.gz', 'rt')):
        cid = r['id'].rsplit('/', 1)[0]; c = cands[cid]
        w = r['generator'].split('/')[-1]
        rows.append({'id': r['id'], 'text': r['text'], 'regions': r['regions'], 'slot': 'mixed', 'split': 'heldout', 'label': 'mixed',
                     'condition': r['edit_type'], 'writer': w})
        orig = c['before'] + '\n\n' + c['target'] + '\n\n' + c['after']
        controls.setdefault(cid, {'id': f'{cid}/original', 'text': orig, 'regions': [{'start': 0, 'end': len(orig), 'label': 0}],
                                  'slot': 'human_controls', 'split': 'heldout', 'label': 'human', 'condition': 'original', 'writer': 'human'})
    with gzip.open(out / 'heldout-score-rows.jsonl.gz', 'wt') as f:
        for x in rows + list(controls.values()):
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(json.dumps({'mixed': len(rows), 'human_controls': len(controls), 'by_writer': dict(Counter(x['writer'] for x in rows))}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('prepare'); p1.add_argument('parquet'); p1.add_argument('out')
    p2 = sub.add_parser('generate'); p2.add_argument('out'); p2.add_argument('--target', type=int, default=200)
    p2.add_argument('--max-calls', type=int, default=320); p2.add_argument('--budget', type=float, default=5.)
    p2.add_argument('--tag', default='pilot'); p2.add_argument('--conc', type=int, default=16)
    p2.add_argument('--passes', type=int, default=1); p2.add_argument('--pass-wait', type=int, default=600)
    p3 = sub.add_parser('prepare-v1'); p3.add_argument('parquet'); p3.add_argument('out'); p3.add_argument('excluded')
    p4 = sub.add_parser('prepare-heldout'); p4.add_argument('parquet'); p4.add_argument('out'); p4.add_argument('eval_ids')
    p4.add_argument('--overlap', default=None)
    p5 = sub.add_parser('claude-batches'); p5.add_argument('out'); p5.add_argument('dest')
    p6 = sub.add_parser('ingest-claude'); p6.add_argument('--writer', required=True, choices=['haiku', 'sonnet', 'opus', 'fable'])
    p6.add_argument('--in', dest='indir', required=True); p6.add_argument('--out', default=str(Path(__file__).resolve().parent))
    p6.add_argument('--split', choices=['heldout', 'train'], default='heldout')
    p8 = sub.add_parser('heldout-fable'); p8.add_argument('--out', default=str(Path(__file__).resolve().parent))
    p9 = sub.add_parser('score-rows'); p9.add_argument('--out', default=str(Path(__file__).resolve().parent))
    p7 = sub.add_parser('claude-train-batches'); p7.add_argument('dest'); p7.add_argument('--out', default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare(a.parquet, Path(a.out))
    elif a.cmd == 'prepare-v1':
        prepare_v1(a.parquet, Path(a.out), a.excluded)
    elif a.cmd == 'prepare-heldout':
        prepare_heldout(a.parquet, Path(a.out), a.eval_ids, a.overlap)
    elif a.cmd == 'claude-batches':
        write_claude_batches(Path(a.out), Path(a.dest))
    elif a.cmd == 'ingest-claude':
        ingest_claude(Path(a.out), a.writer, a.indir, a.split)
    elif a.cmd == 'heldout-fable':
        prepare_heldout_fable(Path(a.out))
    elif a.cmd == 'score-rows':
        build_score_rows(Path(a.out))
    elif a.cmd == 'claude-train-batches':
        prepare_claude_train(Path(a.out), Path(a.dest))
    else:
        asyncio.run(generate(Path(a.out), a.target, a.max_calls, a.budget, a.conc, a.tag, a.passes, a.pass_wait))
