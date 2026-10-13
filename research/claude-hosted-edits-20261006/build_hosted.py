"""Jobs D and E: Claude edits hosted in approved human-source-mix scientific passages (pmc, pes2o, arxiv, acl; dated <=2022).

D (sentence edits, training only): rewrite_one / rewrite_two / insert_one ~30% each, split_one ~10%, same prompts and
   aligner as research/llm-sentence-edits-20261005 (claude-v1), rows -> llm-edits-claude-hsm-v1.jsonl.gz.
E (whole-paragraph edits): 50/50 'draft' (write the paragraph from the surrounding paragraphs without seeing it) and
   'rewrite' (polish the paragraph); the whole paragraph is one AI span; 15% held out (never-train list).
Each host paragraph is used by one job only. Sizing: AI output characters per job ~= Job B body characters
(300 x median synthetic-research-papers-600 body = 9.03M characters).

Usage:
  build_hosted.py candidates HSM_DATA_DIR EXCLUDE_IDS_JSON         # -> hosts-raw.jsonl + snippets for the H200 check
  build_hosted.py prepare --overlap OVERLAP_JSON                    # assign D/E, writers, held-out; write batches
  build_hosted.py ingest-d SCRATCH [--writer W] | ingest-e SCRATCH [--writer W]
"""
import argparse, os, gzip, hashlib, json, random, re, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder of the Oct 6 session that made the intermediate files (outside the repo); set SCRATCH to it

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EDITS = ROOT / 'research/llm-sentence-edits-20261005'
SUITE = ROOT / 'research/evaluation/space-suite-v1/package'
sys.path.insert(0, str(EDITS))
SOURCES = ['pmc', 'pes2o', 'arxiv', 'acl']
B_CHARS = 9_031_050
MODEL = {'sonnet': 'claude-sonnet-5-5', 'opus': 'claude-opus-5-5', 'haiku': 'claude-haiku-4-5'}
ASSIGN_RATIO = {'sonnet': 2.5, 'opus': 2.5, 'haiku': 1.5}  # Haiku over-assigned 1.5x for its lower acceptance
HELDOUT_FRAC_E = .15


def split_paragraphs(text):
    out = []
    for block in re.split(r'\n\s*\n', text):
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        if len(lines) > 1 and statistics.median(len(l) for l in lines) < 110:
            out.append(re.sub(r'(\w)-\s+(?=[a-z])', r'\1', ' '.join(lines)))  # hard-wrapped block: one paragraph
        else:
            out.extend(lines)
    return [re.sub(r'\s+', ' ', p).strip() for p in out]


def host_ok(p):
    from build_llm_edits import sentences, mathy, reference_like
    return (350 <= len(p) <= 1600 and p[:1].isupper() and re.search(r'[.!?]["”)]?$', p) and len(sentences(p)) >= 3
            and '�' not in p and not mathy(p) and not reference_like(p) and not re.match(r'(Figure|Fig\.|Table)\s*\d', p)
            and p.count('$') <= 4)


def candidates(data_dir, exclude_json):
    import pyarrow.parquet as pq
    from build_llm_edits import never_train_hit
    ex = set(json.loads(Path(exclude_json).read_text())['paper_ids'])
    stats = Counter(); hosts = []
    for s in SOURCES:
        for r in pq.read_table(Path(data_dir) / 'data' / f'{s}.parquet', columns=['record_id', 'text', 'claimed_original_date', 'parent_document_id', 'source_url']).to_pylist():
            y = re.match(r'(\d{4})', str(r['claimed_original_date'] or ''))
            if y and int(y.group(1)) > 2022:
                stats['after_2022'] += 1; continue
            ps = split_paragraphs(r['text'])
            for i, p in enumerate(ps):
                if not host_ok(p):
                    continue
                h = {'host_id': f"{s}:{r['record_id'][:16]}/p{i:02d}", 'record_id': r['record_id'], 'source': s, 'year': int(y.group(1)) if y else None,
                     'parent_document_id': r['parent_document_id'], 'prev': ps[i - 1] if i > 0 else '', 'target': p,
                     'next': ps[i + 1] if i + 1 < len(ps) else ''}
                if r['record_id'] in ex or never_train_hit({'paper_id': r['record_id'], 'text': p}):
                    stats['never_train_or_excluded'] += 1; continue
                hosts.append(h)
    with open(HERE / 'hosts-raw.jsonl', 'w') as f:
        for h in hosts:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    sn = {h['host_id']: [y for n in [re.sub(r'\W+', '', h['target'].lower())] for y in (n[:60], n[len(n) // 2:len(n) // 2 + 60]) if len(y) == 60] for h in hosts}
    (SCRATCH / 'hsm/host-snippets.json').write_text(json.dumps(sn))
    stats['hosts'] = len(hosts); stats.update({f'hosts_{k}': v for k, v in Counter(h['source'] for h in hosts).items()})
    stats['passages_with_hosts'] = len({h['record_id'] for h in hosts})
    print(json.dumps(stats, indent=1))


# ---------- prepare ----------

ACCEPT = {'sonnet': .9, 'opus': .9, 'haiku': .61}  # expected acceptance (Haiku from the held-out/claude-v1 ingests)
C13 = ['sonnet', 'opus', 'haiku', 'sonnet', 'opus', 'sonnet', 'opus', 'haiku', 'sonnet', 'opus', 'sonnet', 'opus', 'haiku']  # 5:5:3
NEVER_TRAIN = HERE / 'heldout-never-train.json'
SCR = (SCRATCH / 'claude-hosted')


def prepare(overlap_json):
    from build_llm_edits import V1_MIX, sentences, claude_instruction
    rng = random.Random(20261006)
    hosts = [json.loads(l) for l in open(HERE / 'hosts-raw.jsonl')]
    ov = json.loads(Path(overlap_json).read_text())
    bad = set(ov['selection_calibration_eval']) | set(ov['space_suite']); in_train = set(ov['training'])
    hosts = [h for h in hosts if h['host_id'] not in bad]
    train_records = {h['record_id'] for h in hosts if h['host_id'] in in_train}
    rng.shuffle(hosts)
    mean_len = statistics.mean(len(h['target']) for h in hosts)
    acc = sum(ASSIGN_RATIO[w] * ACCEPT[w] for w in ASSIGN_RATIO) / sum(ASSIGN_RATIO.values())
    need = round(B_CHARS / (mean_len * acc))  # hosts per job to reach ~B characters of accepted AI output
    per_job = min(need, len(hosts) // 2)
    # E first: held-out E items need hosts whose passage never appears in training data
    clean = [h for h in hosts if h['record_id'] not in train_records]; dirty = [h for h in hosts if h['record_id'] in train_records]
    n_hold = round(HELDOUT_FRAC_E * per_job)
    # held-out: whole passages (all their hosts go to held-out E or are left unused), stratified by source
    by_rec = defaultdict(list)
    for h in clean:
        by_rec[h['record_id']].append(h)
    recs = sorted(by_rec, key=lambda r: rng.random()); held, used = [], set()
    src_target = {s_: n_hold * sum(1 for h in hosts if h['source'] == s_) / len(hosts) for s_ in SOURCES}
    got = Counter()
    for r in recs:
        s_ = by_rec[r][0]['source']
        if len(held) >= n_hold or got[s_] >= src_target[s_] + 1:
            continue
        held.append(by_rec[r][0]); used.add(r); got[s_] += 1
    # register held-out passages now, then drop every remaining host that the never-train lists flag (duplicated
    # passages, e.g. the same article under several peS2o/PMC records)
    from build_llm_edits import never_train_hit
    if NEVER_TRAIN.exists():
        NEVER_TRAIN.unlink()
    write_never_train(sorted({h['record_id'] for h in held}), [x for h in held for x in (h['prev'], h['target'], h['next']) if x])
    never_train_hit.__defaults__[0].clear()
    rest = [h for h in hosts if h['record_id'] not in used and not never_train_hit(
        {'paper_id': h['record_id'], 'text': '\n\n'.join(x for x in (h['prev'], h['target'], h['next']) if x)})]
    per_job = min(need, (len(rest) + len(held)) // 2)  # split the remaining supply evenly between E and D
    e_train = rest[:per_job - len(held)]; d_hosts = rest[per_job - len(held):per_job - len(held) + per_job]
    e_items = [dict(h, split='heldout') for h in held] + [dict(h, split='train') for h in e_train]
    rng.shuffle(e_items)
    for k, h in enumerate(e_items):
        h['kind'] = 'draft' if k % 2 == 0 else 'rewrite'
    for kind in ('draft', 'rewrite'):
        for i, h in enumerate(sorted([h for h in e_items if h['kind'] == kind], key=lambda h: (h['split'] == 'train', rng.random()))):
            h['writer'] = C13[i % len(C13)]
    d_items = [dict(h, split='train') for h in d_hosts]
    for i, h in enumerate(d_items):
        h['writer'] = C13[i % len(C13)]
    cnt = Counter()
    for h in d_items:
        h['edit_type'] = V1_MIX[cnt[h['writer']] % len(V1_MIX)]; cnt[h['writer']] += 1
        h['cand_id'] = h['host_id']
    with open(HERE / 'items-d.jsonl', 'w') as f:
        for h in d_items:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    with open(HERE / 'items-e.jsonl', 'w') as f:
        for h in e_items:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    for job, items in (('d', d_items), ('e', e_items)):
        bd = SCR / f'{job}-batches'; bd.mkdir(parents=True, exist_ok=True)
        for w in ASSIGN_RATIO:
            xs = [h for h in items if h['writer'] == w]
            for b in range(0, len(xs), 50):
                with open(bd / f'{w}-{b // 50 + 1:03d}.jsonl', 'w') as f:
                    for h in xs[b:b + 50]:
                        if job == 'd':
                            c = {'cand_id': h['host_id'], 'target': h['target'], 'edit_type': h['edit_type']}
                            rec = {'id': h['host_id'], 'edit_type': h['edit_type'], 'instruction': claude_instruction(c),
                                   'prev_context': h['prev'], 'paragraph': h['target'],
                                   'numbered_sentences': [f'[{i + 1}] ' + h['target'][x:y] for i, (x, y) in enumerate(sentences(h['target']))],
                                   'next_context': h['next']}
                        else:
                            rec = {'id': h['host_id'], 'kind': h['kind'], 'prev_context': h['prev'], 'next_context': h['next'],
                                   'target_words': len(h['target'].split()), 'target_sentences': len(sentences(h['target']))}
                            if h['kind'] == 'rewrite':
                                rec['paragraph'] = h['target']
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    hold = [h for h in e_items if h['split'] == 'heldout']
    exp_chars = lambda items: round(sum(len(h['target']) * ACCEPT[h['writer']] for h in items))
    out = {'hosts_usable': len(hosts), 'hosts_in_training_windows': len(in_train), 'mean_host_chars': round(mean_len),
           'b_chars_target': B_CHARS, 'hosts_needed_per_job': need, 'hosts_assigned_per_job': per_job,
           'supply_fraction_of_target': round(per_job / need, 3),
           'D': {'assigned': len(d_items), 'by_writer': dict(Counter(h['writer'] for h in d_items)),
                 'by_type': dict(Counter(h['edit_type'] for h in d_items)), 'by_source': dict(Counter(h['source'] for h in d_items)),
                 'expected_accepted_chars': exp_chars(d_items), 'batch_files': {w: -(-sum(h['writer'] == w for h in d_items) // 50) for w in ASSIGN_RATIO}},
           'E': {'assigned': len(e_items), 'heldout': len(hold), 'by_writer_kind': {w: dict(Counter(h['kind'] for h in e_items if h['writer'] == w)) for w in ASSIGN_RATIO},
                 'by_writer_split': {w: dict(Counter(h['split'] for h in e_items if h['writer'] == w)) for w in ASSIGN_RATIO},
                 'by_source_split': {s_: dict(Counter(h['split'] for h in e_items if h['source'] == s_)) for s_ in SOURCES},
                 'expected_accepted_chars': exp_chars(e_items), 'batch_files': {w: -(-sum(h['writer'] == w for h in e_items) // 50) for w in ASSIGN_RATIO}},
           'unused_hosts': len(hosts) - len(d_items) - len(e_items), 'dropped_never_train_after_holdout': len(hosts) - len(held) - len(rest) - sum(len(by_rec[r]) - 1 for r in used)}
    (HERE / 'prepare-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def write_never_train(ids, paras):
    old = json.loads(NEVER_TRAIN.read_text()) if NEVER_TRAIN.exists() else {'paper_ids': [], 'paragraph_sha256': [], 'paragraph_shingle_sha256_12': []}
    nh = lambda t: hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()
    hs = {nh(x) for x in paras if x.strip()}
    sh = {hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] for x in paras for n in [re.sub(r'\W+', '', x.lower())] for i in range(0, len(n) - 59, 10)}
    NEVER_TRAIN.write_text(json.dumps({
        'purpose': 'Held-out Job E whole-paragraph items (claude-hosted E split=heldout) and their host passages. Never train on these.',
        'updated_pdt': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
        'check': 'build_llm_edits.never_train_hit(row) loads this file together with the other held-out lists.',
        'normalization': "re.sub(r'\\W+', '', text.lower())",
        'paper_ids': sorted(set(old['paper_ids']) | set(ids)), 'paragraph_sha256': sorted(set(old['paragraph_sha256']) | hs),
        'paragraph_shingle_sha256_12': sorted(set(old['paragraph_shingle_sha256_12']) | sh)}, separators=(',', ':')))


# ---------- ingest ----------

def read_outputs(d, key):
    got, bad = {}, Counter()  # malformed lines per writer folder
    for f in sorted(Path(d).glob('*/*.jsonl')):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                x = json.loads(l)
            except ValueError:
                bad[f.parent.name] += 1; continue
            if x.get('id') and x['id'] not in got:
                got[x['id']] = x.get(key, '')
    return got, bad


def hosted_text(h, mid):
    parts = [x for x in (h['prev'], mid, h['next']) if x]
    text = '\n\n'.join(parts); s0 = len(h['prev']) + 2 if h['prev'] else 0
    return text, s0


def write_rows(name, rows, writer, items):
    path = HERE / name
    old = [json.loads(l) for l in gzip.open(path, 'rt')] if path.exists() and writer else []
    mine = {h['host_id'] for h in items.values() if h['writer'] == writer} if writer else set()
    old = [r for r in old if r['host_id'] not in mine]
    with gzip.open(path, 'wt') as f:
        for r in old + rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def load_items(job):
    return {h['host_id']: h for f in (f'items-{job}.jsonl', f'items-{job}-gap.jsonl', f'items-{job}-scale.jsonl', f'items-{job}-scale2.jsonl', f'items-{job}-scale3.jsonl', f'items-{job}-wide1.jsonl') if (HERE / f).exists() for h in map(json.loads, open(HERE / f))}


def collect_outputs(job, scratch):
    key = 'edited_paragraph' if job == 'd' else 'paragraph'
    got, badj = read_outputs(Path(scratch) / f'{job}-outputs', key)
    g2, b2 = read_outputs(HANDOFF / 'outputs' / job, key)
    return {**g2, **got}, badj + b2


def bad_lines(badj, writer):
    return badj.get(writer, 0) if writer else sum(badj.values())


def check_d(h, edited):
    """Shared by ingest-d and selfcheck.py: strict named-sentence validation + never-train. -> (row, None) or (None, reason)."""
    from build_llm_edits import strict_edit, plan, never_train_hit
    a, b = plan({'target': h['target'], 'edit_type': h['edit_type']}, random.Random(h['host_id']))
    res, reason = strict_edit(h['target'], edited, h['edit_type'], a, b)
    if not res:
        return None, reason
    o0, o1, new = res; t = h['target']
    if o0 == o1:
        new_t = t[:o0] + (' ' if o0 else '') + new + ('' if o0 else ' ') + t[o1:]; s_in = o0 + (1 if o0 else 0)
    else:
        new_t = t[:o0] + new + t[o1:]; s_in = o0
    text, ts = hosted_text(h, new_t); s0 = ts + s_in; s1 = s0 + len(new)
    assert text[s0:s1] == new
    row = {'id': f"{h['host_id']}/{h['writer']}-{h['edit_type']}", 'host_id': h['host_id'], 'paper_id': h['record_id'], 'group': 'human:' + h['record_id'],
           'dataset': 'hsm_llm_edit', 'text': text, 'target_start': ts, 'target_end': ts + len(new_t),
           'regions': [r for r in ({'start': 0, 'end': s0, 'label': 0}, {'start': s0, 'end': s1, 'label': 1}, {'start': s1, 'end': len(text), 'label': 0}) if r['end'] > r['start']],
           'edit_size': 'one' if h['edit_type'] in ('rewrite_one', 'insert_one') else 'two', 'edit_type': h['edit_type'],
           'replaced_human': t[o0:o1], 'inserted_ai': new, 'generator': MODEL[h['writer']], 'written_by': WRITTEN_BY[h['writer']],
           'source': h['source'], 'year': h['year'], 'split': 'train'}
    if never_train_hit(row):
        return None, 'never_train_hit'
    return row, None


def check_e(h, para):
    """Shared by ingest-e and selfcheck.py. Only the target paragraph may be written; context is never echoed.
    -> ([ai_row, human_row], None) or (None, reason)."""
    from build_llm_edits import never_train_hit, sim
    if not para or not para.strip():
        return None, 'empty'
    if re.search(r'\n\s*\n', para.strip()):
        return None, 'not_one_paragraph'
    para = re.sub(r'\s+', ' ', para).strip()
    if para.startswith('#') or '**' in para or re.match(r'(Here is|Sure|Below is)\b', para):
        return None, 'markdown_or_commentary'
    for side in ('prev', 'next'):
        c = h[side]
        cn = re.sub(r'\W+', '', (c or '').lower())  # substring test only for real paragraphs: scrap lines ('4', 'F') match anything
        if c and (sim(c, para) > .6 or (len(cn) >= 40 and cn[:80] in re.sub(r'\W+', '', para.lower()))):
            return None, f'copies {side}_context'
    r_ = len(para.split()) / max(1, len(h['target'].split()))
    if not .5 <= r_ <= 1.8:
        return None, 'length_out_of_range'
    s_ = sim(h['target'], para)
    if s_ > .9:
        return None, 'too_close_to_original'
    if h['kind'] == 'rewrite' and s_ < .12:
        return None, 'rewrite_lost_content'
    text, s0 = hosted_text(h, para); s1 = s0 + len(para)
    htext, hs0 = hosted_text(h, h['target'])
    base = {'host_id': h['host_id'], 'paper_id': h['record_id'], 'group': 'human:' + h['record_id'], 'item_kind': h['kind'], 'source': h['source'],
            'year': h['year'], 'split': h['split']}
    row = {'id': f"{h['host_id']}/{h['writer']}-{h['kind']}", **base, 'dataset': 'hsm_paragraph_edit', 'text': text, 'target_start': s0, 'target_end': s1,
           'regions': [r for r in ({'start': 0, 'end': s0, 'label': 0}, {'start': s0, 'end': s1, 'label': 1}, {'start': s1, 'end': len(text), 'label': 0}) if r['end'] > r['start']],
           'generator': MODEL[h['writer']], 'written_by': WRITTEN_BY[h['writer']], 'similarity_to_original': round(s_, 3)}
    if h['split'] == 'train' and never_train_hit(row):
        return None, 'never_train_hit'
    human = {'id': f"{h['host_id']}/human", **base, 'dataset': 'hsm_paragraph_edit_human', 'text': htext, 'target_start': hs0,
             'target_end': hs0 + len(h['target']), 'regions': [{'start': 0, 'end': len(htext), 'label': 0}], 'generator': None, 'matched_id': row['id']}
    return [row, human], None


def report(job, items, got, rows, rej, writer):
    mine = [h for h in items.values() if not writer or h['writer'] == writer]
    ret = sum(1 for h in mine if h['host_id'] in got)
    return {'job': job.upper(), 'writer': writer or 'all', 'assigned': len(mine), 'returned': ret, 'accepted': len(rows),
            'acceptance_of_returned': round(len(rows) / max(1, ret), 3), 'acceptance_of_assigned': round(len(rows) / max(1, len(mine)), 3),
            'rejects': dict(rej)}


def ingest_d(scratch, writer=None):
    items = load_items('d'); got, badj = collect_outputs('d', scratch)
    nb = bad_lines(badj, writer); rej = Counter({'bad_json_line': nb}) if nb else Counter(); rows = []
    for hid, edited in got.items():
        h = items.get(hid)
        if not h or (writer and h['writer'] != writer):
            continue
        row, reason = check_d(h, edited)
        if reason:
            rej[reason] += 1; continue
        rows.append(row)
    write_rows('llm-edits-claude-hsm-v1.jsonl.gz', rows, writer, items)
    print(json.dumps({**report('d', items, got, rows, rej, writer), 'ai_chars': sum(len(r['inserted_ai']) for r in rows)}, indent=1))


def ingest_e(scratch, writer=None):
    items = load_items('e'); got, badj = collect_outputs('e', scratch)
    nb = bad_lines(badj, writer); rej = Counter({'bad_json_line': nb}) if nb else Counter(); rows, held = [], {}
    for hid, para in got.items():
        h = items.get(hid)
        if not h or (writer and h['writer'] != writer):
            continue
        pair, reason = check_e(h, para)
        if reason:
            rej[reason] += 1; continue
        rows.extend(pair)
        if h['split'] == 'heldout':
            held[hid] = pair[0]['text'][pair[0]['target_start']:pair[0]['target_end']]
    for split in ('train', 'heldout'):
        write_rows(f'paragraph-edits-claude-hsm-v1-{split}.jsonl.gz', [r for r in rows if r['split'] == split], writer, items)
    if held:
        write_never_train([], list(held.values()))
    ai = [r for r in rows if r['generator']]
    print(json.dumps({**report('e', items, got, ai, rej, writer), 'heldout': len(held), 'by_kind': dict(Counter(r['item_kind'] for r in ai)),
                      'ai_chars': sum(r['target_end'] - r['target_start'] for r in ai)}, indent=1))

# ---------- per-writer gap fill (Oct 6 morning) ----------

MODEL.update({'gpt-6.1-sol': 'openai/gpt-6.1-sol', 'gpt-6-sol': 'openai/gpt-6-sol'})
WRITTEN_BY = {'sonnet': 'claude-code-subagent', 'opus': 'claude-code-subagent', 'haiku': 'claude-code-subagent',
              'gpt-6.1-sol': 'codex-subagent', 'gpt-6-sol': 'codex-subagent'}
HANDOFF = ROOT / 'research/openai-writer-handoff-20261006'
GAP = {'d': {'sonnet': 75, 'opus': 75, 'gpt-6.1-sol': 500, 'gpt-6-sol': 500},
       'e': {'sonnet': 650, 'opus': 650, 'haiku': 300, 'gpt-6.1-sol': 210, 'gpt-6-sol': 210}}


def relaxed_ok(p):
    from build_llm_edits import sentences, mathy, reference_like
    return (220 <= len(p) <= 2600 and p[:1].isupper() and re.search(r'[.!?]["”)]?$', p) and len(sentences(p)) >= 2
            and '�' not in p and not mathy(p) and not reference_like(p) and not re.match(r'(Figure|Fig\.|Table)\s*\d', p) and p.count('$') <= 4)


def gap_hosts(data_dir, free_json):
    """Unused clean hosts from hosts-raw.jsonl plus extra paragraphs admitted by relaxed length/sentence bounds."""
    import pyarrow.parquet as pq
    from build_llm_edits import never_train_hit
    free = set(json.loads(Path(free_json).read_text()))
    raw = {json.loads(l)['host_id']: json.loads(l) for l in open(HERE / 'hosts-raw.jsonl')}
    out = [dict(raw[h], admission='standard') for h in free]; stats = Counter(standard=len(out))
    for s_ in SOURCES:
        for r in pq.read_table(Path(data_dir) / 'data' / f'{s_}.parquet', columns=['record_id', 'text', 'claimed_original_date', 'parent_document_id']).to_pylist():
            y = re.match(r'(\d{4})', str(r['claimed_original_date'] or ''))
            if y and int(y.group(1)) > 2022:
                continue
            ps = split_paragraphs(r['text'])
            for i, p in enumerate(ps):
                hid = f"{s_}:{r['record_id'][:16]}/p{i:02d}"
                if hid in raw or not relaxed_ok(p):
                    continue
                h = {'host_id': hid, 'record_id': r['record_id'], 'source': s_, 'year': int(y.group(1)) if y else None,
                     'parent_document_id': r['parent_document_id'], 'prev': ps[i - 1] if i > 0 else '', 'target': p,
                     'next': ps[i + 1] if i + 1 < len(ps) else '', 'admission': 'relaxed_bounds'}
                if never_train_hit({'paper_id': r['record_id'], 'text': '\n\n'.join(x for x in (h['prev'], p, h['next']) if x)}):
                    stats['never_train'] += 1; continue
                out.append(h); stats['relaxed'] += 1
    with open(HERE / 'hosts-gap-raw.jsonl', 'w') as f:
        for h in out:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    sn = {h['host_id']: [y for n in [re.sub(r'\W+', '', h['target'].lower())] for y in (n[:60], n[len(n) // 2:len(n) // 2 + 60]) if len(y) == 60] for h in out if h['admission'] != 'standard'}
    (SCRATCH / 'hsm/gap-snippets.json').write_text(json.dumps(sn))
    print(dict(stats), len(out))


def prepare_gap(overlap_json):
    from build_llm_edits import V1_MIX, sentences, claude_instruction
    rng = random.Random(20261008)
    ov = json.loads(Path(overlap_json).read_text()); bad = set(ov['selection_calibration_eval']) | set(ov['space_suite'])
    hosts = [h for h in map(json.loads, open(HERE / 'hosts-gap-raw.jsonl')) if h['host_id'] not in bad]
    rng.shuffle(hosts)
    hosts.sort(key=lambda h: h['admission'] != 'standard')  # standard-bound hosts first
    d_ok = [h for h in hosts if len(sentences(h['target'])) >= 3 and len(h['target']) <= 1600]
    used = set(); items = {'d': [], 'e': []}
    for w, n in GAP['d'].items():
        xs = [h for h in d_ok if h['host_id'] not in used][:n]
        for k, h in enumerate(xs):
            used.add(h['host_id']); items['d'].append(dict(h, writer=w, split='train', edit_type=V1_MIX[k % len(V1_MIX)], cand_id=h['host_id']))
    for w, n in GAP['e'].items():
        xs = [h for h in hosts if h['host_id'] not in used][:n]
        for k, h in enumerate(xs):
            used.add(h['host_id']); items['e'].append(dict(h, writer=w, split='train', kind='draft' if k % 2 == 0 else 'rewrite'))
    short = {job: {w: n - sum(h['writer'] == w for h in items[job]) for w, n in GAP[job].items()} for job in GAP}
    for job in ('d', 'e'):
        with open(HERE / f'items-{job}-gap.jsonl', 'w') as f:
            for h in items[job]:
                f.write(json.dumps(h, ensure_ascii=False) + '\n')
        for w in GAP[job]:
            xs = [h for h in items[job] if h['writer'] == w]
            bd = (HANDOFF / 'batches' / job) if w.startswith('gpt') else (SCR / f'{job}-gap')
            bd.mkdir(parents=True, exist_ok=True)
            name = (lambda b: f'{w}-{b:03d}.jsonl') if w.startswith('gpt') else (lambda b: f'gap-{w}-{b:03d}.jsonl')
            for b in range(0, len(xs), 50):
                with open(bd / name(b // 50 + 1), 'w') as f:
                    for h in xs[b:b + 50]:
                        f.write(json.dumps(batch_rec(job, h), ensure_ascii=False) + '\n')
    out = {'hosts_available': len(hosts), 'hosts_standard': sum(h['admission'] == 'standard' for h in hosts), 'shortfall': short,
           **{job.upper(): {'items': len(items[job]), 'by_writer': dict(Counter(h['writer'] for h in items[job])),
                            'by_admission': dict(Counter(h['admission'] for h in items[job])),
                            'by_source': dict(Counter(h['source'] for h in items[job])),
                            'by_type' if job == 'd' else 'by_writer_kind': (dict(Counter(h['edit_type'] for h in items[job])) if job == 'd'
                                                                            else {w: dict(Counter(h['kind'] for h in items[job] if h['writer'] == w)) for w in GAP['e']})}
              for job in ('d', 'e')}}
    (HERE / 'prepare-gap-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def batch_rec(job, h):
    from build_llm_edits import sentences, claude_instruction
    if job == 'd':
        c = {'cand_id': h['host_id'], 'target': h['target'], 'edit_type': h['edit_type']}
        return {'id': h['host_id'], 'edit_type': h['edit_type'], 'instruction': claude_instruction(c), 'prev_context': h['prev'],
                'paragraph': h['target'], 'numbered_sentences': [f'[{i + 1}] ' + h['target'][x:y] for i, (x, y) in enumerate(sentences(h['target']))],
                'next_context': h['next']}
    rec = {'id': h['host_id'], 'kind': h['kind'], 'prev_context': h['prev'], 'next_context': h['next'],
           'target_words': len(h['target'].split()), 'target_sentences': len(sentences(h['target']))}
    if h['kind'] == 'rewrite':
        rec['paragraph'] = h['target']
    return rec


# ---------- T2 scale-up (Oct 6 afternoon) ----------
# The original Job D/E items (items-d.jsonl / items-e.jsonl, 9,551 hosts each) were never written (D/E were cancelled).
# Their training hosts are re-assigned here; the unrun item lists are kept as items-{d,e}.unrun.jsonl for provenance.
SCALE = {'d': {'sonnet': 3600, 'opus': 3530, 'haiku': 1380, 'gpt-6.1-sol': 780, 'gpt-6-sol': 780},
         'e': {'sonnet': 750, 'opus': 750, 'haiku': 150, 'gpt-6.1-sol': 210, 'gpt-6-sol': 210}}
# accepted-row targets: Claude D 7,600 (S 3,420 / O 3,420 / H 760), Claude E 1,500 (S 712 / O 713 / H 75), GPT D 1,500, GPT E 400.
# Assignments include margin for strict-ingest acceptance measured on Oct 6 (D: Sonnet 97%, Opus 99%, Haiku 58%;
# E: Sonnet 98%, Opus 96%, Haiku 53%; GPT round 1: 100% D, 99.5% E).


def prepare_scale():
    from build_llm_edits import V1_MIX, sentences, never_train_hit
    rng = random.Random(20261010)
    unrun = {}
    for job in ('d', 'e'):
        src = HERE / f'items-{job}.jsonl'; dst = HERE / f'items-{job}.unrun.jsonl'
        if src.exists() and not dst.exists():
            src.rename(dst)
        unrun[job] = [json.loads(l) for l in open(dst)]
    taken = {h['host_id'] for job in ('d', 'e') for f in (f'items-{job}-gap.jsonl',) for h in map(json.loads, open(HERE / f))}
    ok = lambda h: h['host_id'] not in taken and not never_train_hit({'paper_id': h['record_id'], 'text': '\n\n'.join(x for x in (h['prev'], h['target'], h['next']) if x)})
    d_pool = [h for h in unrun['d'] if ok(h)]
    e_pool = [h for h in unrun['e'] if h['split'] == 'train' and ok(h)]
    gap_raw = [json.loads(l) for l in open(HERE / 'hosts-gap-raw.jsonl')]
    ov = json.loads((SCRATCH / 'hsm/gap-overlap.json').read_text())
    bad = set(ov['selection_calibration_eval']) | set(ov['space_suite'])
    extra = [h for h in gap_raw if h['host_id'] not in bad and ok(h)]
    rng.shuffle(d_pool); rng.shuffle(e_pool); rng.shuffle(extra)
    d_extra = [h for h in extra if len(sentences(h['target'])) >= 3 and len(h['target']) <= 1600]
    used = set(); items = {'d': [], 'e': []}; short = {}
    e_tail = e_pool[2100:]  # E needs ~2,070 hosts; the rest of the unrun E training hosts can back-fill D
    for job, pools in (('d', [d_pool, d_extra, e_tail]), ('e', [e_pool, extra])):
        for w, n in SCALE[job].items():
            got = []
            for pool in pools:
                for h in pool:
                    if len(got) >= n:
                        break
                    if h['host_id'] not in used:
                        used.add(h['host_id']); got.append(h)
            short[f'{job}/{w}'] = n - len(got)
            for k, h in enumerate(got):
                base = {kk: h[kk] for kk in ('host_id', 'record_id', 'source', 'year', 'parent_document_id', 'prev', 'target', 'next')}
                base['admission'] = h.get('admission', 'standard')
                if job == 'd':
                    items['d'].append({**base, 'writer': w, 'split': 'train', 'edit_type': V1_MIX[k % len(V1_MIX)], 'cand_id': h['host_id'], 'round': 'scale'})
                else:
                    items['e'].append({**base, 'writer': w, 'split': 'train', 'kind': 'draft' if k % 2 == 0 else 'rewrite', 'round': 'scale'})
    for job in ('d', 'e'):
        with open(HERE / f'items-{job}-scale.jsonl', 'w') as f:
            for h in items[job]:
                f.write(json.dumps(h, ensure_ascii=False) + '\n')
        for w in SCALE[job]:
            xs = [h for h in items[job] if h['writer'] == w]
            gpt = w.startswith('gpt')
            bd = (HANDOFF / 'batches' / job) if gpt else (SCR / f'{job}-scale'); bd.mkdir(parents=True, exist_ok=True)
            for b in range(0, len(xs), 50):
                name = f'{w}-r2-{b // 50 + 1:03d}.jsonl' if gpt else f'scale-{w}-{b // 50 + 1:03d}.jsonl'
                with open(bd / name, 'w') as f:
                    for h in xs[b:b + 50]:
                        f.write(json.dumps(batch_rec(job, h), ensure_ascii=False) + '\n')
    out = {'pool_unrun_d_clean': len(d_pool), 'pool_unrun_e_train_clean': len(e_pool), 'pool_extra_relaxed': len(extra),
           'shortfall': {k: v for k, v in short.items() if v}, **{job.upper(): {
               'items': len(items[job]), 'by_writer': dict(Counter(h['writer'] for h in items[job])),
               'by_admission': dict(Counter(h['admission'] for h in items[job])), 'by_source': dict(Counter(h['source'] for h in items[job])),
               'mix': dict(Counter(h['edit_type'] if job == 'd' else h['kind'] for h in items[job])),
               'batch_files': {w: -(-sum(h['writer'] == w for h in items[job]) // 50) for w in SCALE[job]}} for job in ('d', 'e')}}
    (HERE / 'prepare-scale-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


MODEL['fable'] = 'claude-fable-5-1'; WRITTEN_BY['fable'] = 'claude-code-subagent'
FABLE_SCR = (SCRATCH / 'claude-fable')


def carve_fable_d(per_writer=630):
    """Reassign the last `per_writer` Sonnet and Opus D-scale items to Fable (1,260 assigned -> ~1,200 accepted).
    Rewrites the Sonnet/Opus scale batches and writes Fable batches to claude-fable/scale/."""
    path = HERE / 'items-d-scale.jsonl'
    items = [json.loads(l) for l in open(path)]
    if any(h['writer'] == 'fable' for h in items):
        raise SystemExit('already carved')
    for w in ('sonnet', 'opus'):
        idx = [i for i, h in enumerate(items) if h['writer'] == w][-per_writer:]
        for i in idx:
            items[i]['writer'] = 'fable'; items[i]['carved_from'] = w
    with open(path, 'w') as f:
        for h in items:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    for w in ('sonnet', 'opus'):
        for old in (SCR / 'd-scale').glob(f'scale-{w}-*.jsonl'):
            old.unlink()
    for w, bd, pref in (('sonnet', SCR / 'd-scale', 'scale-sonnet'), ('opus', SCR / 'd-scale', 'scale-opus'), ('fable', FABLE_SCR / 'scale', 'fable-d')):
        bd.mkdir(parents=True, exist_ok=True)
        xs = [h for h in items if h['writer'] == w]
        for b in range(0, len(xs), 50):
            with open(bd / f'{pref}-{b // 50 + 1:03d}.jsonl', 'w') as f:
                for h in xs[b:b + 50]:
                    f.write(json.dumps(batch_rec('d', h), ensure_ascii=False) + '\n')
    print(json.dumps({'by_writer': dict(Counter(h['writer'] for h in items)), 'fable_types': dict(Counter(h['edit_type'] for h in items if h['writer'] == 'fable'))}))


# ---------- Claude small-edit scale-up 2 (Oct 6 night): fill edit_claude to 3 full-length epochs (13,320 rows) ----------
SCALE2 = {'opus': 1600, 'fable': 1600}


def prepare_scale2():
    """D items on hosts no earlier items file used (hosts-raw.jsonl, then gap hosts within D bounds), Opus and Fable only."""
    from build_llm_edits import V1_MIX, sentences, never_train_hit
    rng = random.Random(20261011)
    used = {h['host_id'] for job in ('d', 'e') for f in (f'items-{job}-gap.jsonl', f'items-{job}-scale.jsonl', f'items-{job}.unrun.jsonl')
            if (HERE / f).exists() for h in map(json.loads, open(HERE / f))
            if not f.endswith('.unrun.jsonl') or h.get('split') != 'train'}  # unrun train hosts were re-assigned by prepare_scale
    used |= {h['host_id'] for job in ('d', 'e') if (HERE / f'items-{job}-scale.jsonl').exists() for h in map(json.loads, open(HERE / f'items-{job}-scale.jsonl'))}
    ok = lambda h: h['host_id'] not in used and not never_train_hit({'paper_id': h['record_id'], 'text': '\n\n'.join(x for x in (h['prev'], h['target'], h['next']) if x)})
    ov = json.loads((SCRATCH / 'hsm/gap-overlap.json').read_text())
    bad = set(ov['selection_calibration_eval']) | set(ov['space_suite'])
    raw = [json.loads(l) for l in open(HERE / 'hosts-raw.jsonl')]
    gap = [json.loads(l) for l in open(HERE / 'hosts-gap-raw.jsonl')]
    pool = [h for h in raw if ok(h) and h['host_id'] not in bad and host_ok(h['target'])]
    extra = [h for h in gap if ok(h) and h['host_id'] not in bad and len(sentences(h['target'])) >= 3 and len(h['target']) <= 1600]
    rng.shuffle(pool); rng.shuffle(extra)
    seen, items, short = set(), [], {}
    for w, n in SCALE2.items():
        got = []
        for h in pool + extra:
            if len(got) >= n:
                break
            if h['host_id'] not in seen:
                seen.add(h['host_id']); got.append(h)
        short[w] = n - len(got)
        for k, h in enumerate(got):
            base = {kk: h[kk] for kk in ('host_id', 'record_id', 'source', 'year', 'parent_document_id', 'prev', 'target', 'next')}
            base['admission'] = h.get('admission', 'standard')
            items.append({**base, 'writer': w, 'split': 'train', 'edit_type': V1_MIX[k % len(V1_MIX)], 'cand_id': h['host_id'], 'round': 'scale2'})
    with open(HERE / 'items-d-scale2.jsonl', 'w') as f:
        for h in items:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    for w in SCALE2:
        bd = (FABLE_SCR / 'scale2') if w == 'fable' else (SCR / 'd-scale2'); bd.mkdir(parents=True, exist_ok=True)
        xs = [h for h in items if h['writer'] == w]
        for b in range(0, len(xs), 50):
            with open(bd / f'scale2-{w}-{b // 50 + 1:03d}.jsonl', 'w') as f:
                for h in xs[b:b + 50]:
                    f.write(json.dumps(batch_rec('d', h), ensure_ascii=False) + '\n')
    out = {'free_raw_clean': len(pool), 'free_gap_clean': len(extra), 'shortfall': {k: v for k, v in short.items() if v},
           'items': len(items), 'by_writer': dict(Counter(h['writer'] for h in items)), 'by_source': dict(Counter(h['source'] for h in items)),
           'mix': dict(Counter(h['edit_type'] for h in items)), 'by_admission': dict(Counter(h['admission'] for h in items))}
    (HERE / 'prepare-scale2-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


# ---------- Scale-up 3 (Oct 7, ~12:15 AM PDT): hosts screened against the T2.1 held-out split before assignment ----------
# Releases never-run items whose hosts were chosen before the split existed: the 1,600 Fable scale2 items (Fable is
# held-out only), the GPT round-2 D items of both models and the GPT-6 Sol round-2 E items (GPT-6 Sol is held-out only).
# Targets: Claude small edits +2,698 accepted rows, GPT-6.1 Sol small edits +1,403 (training needs for 3 full epochs).
SCALE3 = {'opus': 1450, 'sonnet': 1450, 'gpt-6.1-sol': 1500}


def prepare_scale3():
    from build_llm_edits import V1_MIX, sentences, never_train_hit, NEVER_TRAIN_FILES
    assert any('t21-heldout' in str(f) for f in NEVER_TRAIN_FILES), 'T2.1 held-out lists missing from never-train files'
    rng = random.Random(20261012); released = Counter()
    for f, drop in (('items-d-scale2.jsonl', lambda h: h['writer'] == 'fable'),
                    ('items-d-scale.jsonl', lambda h: h['writer'].startswith('gpt-')),
                    ('items-e-scale.jsonl', lambda h: h['writer'] == 'gpt-6-sol')):
        rows = [json.loads(l) for l in open(HERE / f)]
        bak = HERE / f.replace('.jsonl', '.before-scale3.jsonl')
        if not bak.exists():
            bak.write_text(''.join(json.dumps(h, ensure_ascii=False) + '\n' for h in rows))
        keep = [h for h in rows if not drop(h)]; released[f] = len(rows) - len(keep)
        with open(HERE / f, 'w') as out:
            for h in keep:
                out.write(json.dumps(h, ensure_ascii=False) + '\n')
    sup = HANDOFF / 'batches' / 'superseded-20261007'
    for job, pat in (('d', 'gpt-*-r2-*.jsonl'), ('e', 'gpt-6-sol-r2-*.jsonl')):
        (sup / job).mkdir(parents=True, exist_ok=True)
        for b in (HANDOFF / 'batches' / job).glob(pat):
            b.rename(sup / job / b.name)
    used = {r['host_id'] for r in map(json.loads, gzip.open(HERE / 'llm-edits-claude-hsm-v1.jsonl.gz', 'rt'))}
    used |= {r['host_id'] for r in map(json.loads, gzip.open(HERE / 'paragraph-edits-claude-hsm-v1-train.jsonl.gz', 'rt'))}
    for job in ('d', 'e'):
        for f in (f'items-{job}-gap.jsonl', f'items-{job}-scale.jsonl', f'items-{job}-scale2.jsonl'):
            if (HERE / f).exists():
                used |= {h['host_id'] for h in map(json.loads, open(HERE / f))}
    used |= {h['host_id'] for h in map(json.loads, open(HERE / 'items-d.unrun.jsonl')) if h.get('split') != 'train'}
    ov = json.loads((SCRATCH / 'hsm/gap-overlap.json').read_text())
    bad = set(ov['selection_calibration_eval']) | set(ov['space_suite'])
    nt = lambda h: never_train_hit({'paper_id': h['record_id'], 'text': '\n\n'.join(x for x in (h['prev'], h['target'], h['next']) if x)})
    stats = Counter()
    def clean(hosts, admit):
        out = []
        for h in hosts:
            if h['host_id'] in used: continue
            if h['host_id'] in bad: stats['eval_overlap'] += 1; continue
            if not admit(h): stats['bounds'] += 1; continue
            if nt(h): stats['never_train'] += 1; continue
            out.append(h)
        return out
    raw = clean([json.loads(l) for l in open(HERE / 'hosts-raw.jsonl')], lambda h: host_ok(h['target']))
    gap = clean([json.loads(l) for l in open(HERE / 'hosts-gap-raw.jsonl')], lambda h: len(sentences(h['target'])) >= 3 and len(h['target']) <= 1600)
    rng.shuffle(raw); rng.shuffle(gap); pool = raw + gap; items = []; k0 = 0
    for w, n in SCALE3.items():
        got = pool[k0:k0 + n]; k0 += len(got)
        for k, h in enumerate(got):
            base = {kk: h[kk] for kk in ('host_id', 'record_id', 'source', 'year', 'parent_document_id', 'prev', 'target', 'next')}
            base['admission'] = h.get('admission', 'standard')
            items.append({**base, 'writer': w, 'split': 'train', 'edit_type': V1_MIX[k % len(V1_MIX)], 'cand_id': h['host_id'], 'round': 'scale3'})
    with open(HERE / 'items-d-scale3.jsonl', 'w') as f:
        for h in items:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    for w in SCALE3:
        gpt = w.startswith('gpt'); bd = (HANDOFF / 'batches' / 'd') if gpt else (SCR / 'd-scale3'); bd.mkdir(parents=True, exist_ok=True)
        xs = [h for h in items if h['writer'] == w]
        for b in range(0, len(xs), 50):
            name = f'{w}-r3-{b // 50 + 1:03d}.jsonl' if gpt else f'scale3-{w}-{b // 50 + 1:03d}.jsonl'
            with open(bd / name, 'w') as f:
                for h in xs[b:b + 50]:
                    f.write(json.dumps(batch_rec('d', h), ensure_ascii=False) + '\n')
    e_left = [h for h in map(json.loads, open(HERE / 'items-e-scale.jsonl')) if h['writer'] == 'gpt-6.1-sol']
    out = {'released': dict(released), 'clean_hosts': {'raw': len(raw), 'gap': len(gap)}, 'rejected_hosts': dict(stats),
           'assigned': dict(Counter(h['writer'] for h in items)), 'clean_hosts_left_unassigned': len(pool) - k0,
           'by_source': dict(Counter(h['source'] for h in items)), 'mix': dict(Counter(h['edit_type'] for h in items)),
           'gpt61_e_r2_items': len(e_left), 'gpt61_e_r2_on_heldout_papers': sum(map(nt, e_left))}
    (HERE / 'prepare-scale3-stats.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


# ---------- Wide pool (Oct 7): hosts sampled from the corpus clean-v2 text (research/corpus-sampling-20261007) ----------
def prepare_wide(job, items_jsonl):
    """make_items.py output (writer, edit_type/kind already assigned) -> items-{job}-wide1.jsonl + batches of 50 in SCR/{job}-wide1.
    Only Claude writers get batch files here; API writers (luna, deepseek) run on the Space with api_writer.py."""
    items = [json.loads(l) for l in open(items_jsonl)]
    path = HERE / f'items-{job}-wide1.jsonl'
    old = [json.loads(l) for l in open(path)] if path.exists() else []
    have = {h['host_id'] for h in old}; items = old + [h for h in items if h['host_id'] not in have]
    with open(path, 'w') as f:
        for h in items: f.write(json.dumps(h, ensure_ascii=False) + '\n')
    bd = SCR / f'{job}-wide1'; bd.mkdir(parents=True, exist_ok=True); batches = Counter()
    for w in sorted({h['writer'] for h in items} & {'opus', 'sonnet', 'haiku'}):
        xs = [h for h in items if h['writer'] == w]
        for b in range(0, len(xs), 50):
            with open(bd / f'wide1-{w}-{b // 50 + 1:03d}.jsonl', 'w') as f:
                for h in xs[b:b + 50]: f.write(json.dumps(batch_rec(job, h), ensure_ascii=False) + '\n')
            batches[w] += 1
    print(json.dumps({'items': len(items), 'by_writer': dict(Counter(h['writer'] for h in items)), 'batches': dict(batches)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p0 = sub.add_parser('candidates'); p0.add_argument('data_dir'); p0.add_argument('exclude')
    p1 = sub.add_parser('prepare'); p1.add_argument('--overlap', required=True)
    W = ['sonnet', 'opus', 'haiku', 'fable', 'gpt-6.1-sol', 'gpt-6-sol']
    p2 = sub.add_parser('ingest-d'); p2.add_argument('scratch', nargs='?', default=str(SCR)); p2.add_argument('--writer', choices=W)
    p3 = sub.add_parser('ingest-e'); p3.add_argument('scratch', nargs='?', default=str(SCR)); p3.add_argument('--writer', choices=W)
    p4 = sub.add_parser('gap-hosts'); p4.add_argument('data_dir'); p4.add_argument('free')
    p5 = sub.add_parser('prepare-gap'); p5.add_argument('--overlap', required=True)
    p6 = sub.add_parser('prepare-scale')
    p7 = sub.add_parser('carve-fable-d')
    p8 = sub.add_parser('prepare-scale2')
    p9 = sub.add_parser('prepare-scale3')
    p10 = sub.add_parser('prepare-wide'); p10.add_argument('job', choices=['d', 'e']); p10.add_argument('items')
    a = ap.parse_args()
    if a.cmd == 'candidates':
        candidates(a.data_dir, a.exclude)
    elif a.cmd == 'prepare':
        prepare(a.overlap)
    elif a.cmd == 'gap-hosts':
        gap_hosts(a.data_dir, a.free)
    elif a.cmd == 'prepare-gap':
        prepare_gap(a.overlap)
    elif a.cmd == 'prepare-scale':
        prepare_scale()
    elif a.cmd == 'prepare-wide':
        prepare_wide(a.job, a.items)
    elif a.cmd == 'prepare-scale3':
        prepare_scale3()
    elif a.cmd == 'prepare-scale2':
        prepare_scale2()
    elif a.cmd == 'carve-fable-d':
        carve_fable_d()
    elif a.cmd == 'ingest-d':
        ingest_d(a.scratch, a.writer)
    else:
        ingest_e(a.scratch, a.writer)
