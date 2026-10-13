"""Assemble the 1,020-token copy of the Oct 7 mix and screen it (Oct 10).

For every row of the 72,000-row stage-2 mix (epoch files concatenated in order, as mixgrid/merge_epochs.py does):
  - pick its extension: section rows (sections_ai, human_section) from pools; other paired_pdf_v1 rows from paired;
    other clean_v2 rows from cleanv2; mirrors, human, fullpapers, claude_fp from pools; GRADTEX and old-host rows unchanged;
  - screen the extended row with mix_build.py's rules (never_train_hit over every never-train list, evaluation / selection /
    calibration paper ids, reserved manuscript families, shared 10+-word sentences), here also against every held-out
    evaluation set in eval-inputs/ (stricter than the original mix);
  - if the extended row fails screening, keep the original 510-version of the row (the mix stays row-for-row).
Writes OUT/stage2-epoch0.jsonl.gz, OUT/manifest.json (stage 1, selection and calibration entries copied unchanged from the
base manifest) and OUT/report.json with per-key extension rates, screening rejects, token lengths and the format check.

  uv run --with transformers --with tokenizers --with numpy python merge_screen.py SCRATCH/longctx BASE_MANIFEST OUT
"""
import collections, glob, gzip, hashlib, io, json, os, re, sys
import numpy as np
from transformers import AutoTokenizer

L = sys.argv[1]; BASE = json.load(open(sys.argv[2])); OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
SC = os.path.join(L, 'screen'); tok = AutoTokenizer.from_pretrained(os.path.join(L, 'tokenizer'))
SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)  # mix_build.py:47-48
def sents(t): return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}
def norm_hash(t): return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()
nt = {'papers': set(), 'hashes': set(), 'shingles': set()}
for f in glob.glob(os.path.join(SC, 'never-train', '*.json')):
    d = json.load(open(f)); nt['papers'] |= set(d['paper_ids']); nt['hashes'] |= set(d['paragraph_sha256']); nt['shingles'] |= set(d.get('paragraph_shingle_sha256_12') or [])
def never_train_hit(row):  # mix_build.py:54-62
    ids = {str(row.get(k, '')).replace('paper:', '').replace('human:', '').split('/')[0] for k in ('paper_id', 'group', 'group_id', 'seed_id', 'item_id')}
    ids |= {x for x in re.split(r'[/]', str(row.get('id', ''))) if x.startswith(('claude-fp-', 'claude-sec-'))} | {str(row.get('id', '')).replace('fullpaper-', '')}
    if ids & nt['papers']: return True
    text = row.get('text', '')
    if any(norm_hash(p) in nt['hashes'] for p in text.split('\n\n') if len(re.sub(r'\W+', '', p)) >= 100): return True
    n = re.sub(r'\W+', '', text.lower())
    return any(hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] in nt['shingles'] for i in range(max(0, len(n) - 59)))
keys, held = set(), set()
evfiles = [os.path.join(SC, 'sweep-eval-rows.jsonl.gz')] + sorted(glob.glob(os.path.join(SC, 'eval-inputs', '*.jsonl.gz'))) + sorted(glob.glob(os.path.join(SC, 'eval-inputs', 'hetero', '**', '*.jsonl.gz'), recursive=True))
for f in evfiles:
    for l in gzip.open(f, 'rt'):
        r = json.loads(l); keys.add(r['id'].split('/')[0]); keys.add(str(r.get('group_id') or '').replace('paper:', '')); keys.add(str(r.get('paper_id') or '').replace('paper:', '')); held |= sents(r.get('text', ''))
for name in ['selection-windows', 'calibration-windows']:
    for l in gzip.open(os.path.join(SC, f'{name}.jsonl.gz'), 'rt'):
        r = json.loads(l); keys.add(str(r.get('paper_id', '')).replace('paper:', '')); keys.add(r['id'].split('/')[0]); held |= sents(r['text'])
keys.discard('')
blocked = set(json.load(open(os.path.join(SC, 'reservation.json')))['exclude_from_training']['source_paper_ids'])
def screen(r):  # mix_build.py:85-107 minus the crop (rows are already within 1,000 tokens)
    if never_train_hit(r): return 'never_train'
    ids = {str(r.get('paper_id', '')).replace('paper:', ''), r['id'].split('/')[0], str(r.get('group', '')).replace('paper:', '').replace('human:', '')}
    if ids & keys: return 'eval_paper_id'
    if ids & blocked: return 'reserved_family'
    if sents(r['text']) & held: return 'eval_shared_sentence'
    return None
ext = {}
for fam in ['paired', 'cleanv2', 'pools']:
    ext[fam] = {json.loads(l)['id']: json.loads(l) for l in gzip.open(os.path.join(L, fam, 'extended-ids.jsonl.gz'), 'rt')}
def choose(r):
    k, f = r['dataset'], r.get('text_format')
    if k in ('sections_ai', 'human_section'): return 'pools'
    if k in ('mirrors', 'human', 'fullpapers', 'claude_fp'): return 'pools'
    if f == 'paired_pdf_v1': return 'paired'
    if f == 'clean_v2': return 'cleanv2'
    return None
rows = [json.loads(l) for e in range(3) for l in gzip.open(os.path.join(L, 'input', f'stage2-epoch{e}.jsonl.gz'), 'rt')]
assert len(rows) == 72000 and len({r['id'] for r in rows}) == 72000
out, stat = [], collections.defaultdict(collections.Counter); toklen = collections.defaultdict(list); added = collections.defaultdict(list)
LAB = lambda r: 'doc-only' if r.get('supervision') == 'document_only' else 'ai' if any(g['label'] == 1 for g in r.get('regions') or []) else 'human'
for r in rows:
    fam = choose(r); e = ext.get(fam, {}).get(r['id']) if fam else None; s = stat[(r['dataset'], r.get('text_format'))]; s['rows'] += 1
    if e is not None:
        why = screen(e)
        if why: s['screen:' + why] += 1; e = None
    if e is not None:
        assert e['text'][e['ctx_before_chars']:e['ctx_before_chars'] + len(r['text'])] == r['text']
        s['extended'] += 1; out.append(e)
        ctx = e['text'][:e['ctx_before_chars']] + '\n' + e['text'][e['ctx_before_chars'] + len(r['text']):]
        added[(r.get('text_format'), LAB(r))].append(ctx)
    else:
        if fam: s['kept_original'] += 1
        out.append(r)
    toklen[(r.get('text_format'), LAB(r))].append(len(tok(out[-1]['text'], add_special_tokens=False)['input_ids']))
mx = max(max(v) for (f, lab), v in toklen.items() if lab != 'doc-only'); assert mx <= 1020, mx  # GRADTEX documents are chunked by the trainer
def feats(t):
    lines = [x for x in t.split('\n') if x.strip()]; n = max(1, len(t))
    return {'short_line_share': sum(len(x) for x in lines if len(x) < 60) / n, 'digit_share': sum(c.isdigit() for c in t) / n,
            'math_share': sum(c in '$\\^_=<>{}' for c in t) / n, 'newline_per_1k': 1000 * t.count('\n') / n,
            'citation_per_1k': 1000 * len(re.findall(r'\[\d+(?:[,–-]\s*\d+)*\]|\(\w[\w .&]+,? \d{4}\)', t)) / n}
fmt = {}
for (f, lab), ts in sorted(added.items()):
    F = [feats(t) for t in ts]
    fmt[f'{f}/{lab}'] = {'n': len(ts), 'added_chars_median': int(np.median([len(t) for t in ts])), **{k: round(float(np.median([x[k] for x in F])), 4) for k in F[0]}}
lens = {f'{f}/{lab}': {'n': len(v), 'median': int(np.median(v)), 'p90': int(np.percentile(v, 90))} for (f, lab), v in sorted(toklen.items())}
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as g:
    for r in out: g.write((json.dumps(r, ensure_ascii=False) + '\n').encode())
raw = gzip.decompress(buf.getvalue()); open(os.path.join(OUT, 'stage2-epoch0.jsonl.gz'), 'wb').write(buf.getvalue())
man = json.loads(json.dumps(BASE)); man['files']['stage2-epoch0'] = {'rows': len(out), 'sha256': hashlib.sha256(raw).hexdigest()}
man['counts']['stage2-epoch0'] = dict(collections.Counter(r['dataset'] for r in out))
man['long_context'] = {'window_tokens': 1020, 'built': '2026-10-10', 'note': 'row-for-row copy of moe-final-1ep stage 2 with real surrounding context; see report.json'}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
tot = collections.Counter(); [tot.update(v) for v in stat.values()]
rep = {'rows': len(out), 'extended': tot['extended'], 'totals': dict(tot), 'per_key': {f'{k}/{f}': dict(v) for (k, f), v in sorted(stat.items())},
       'tokens_by_format_and_label': lens, 'added_context_format_check': fmt, 'max_tokens': int(mx), 'screen_inputs': {'never_train_papers': len(nt['papers']), 'eval_keys': len(keys), 'held_sentences': len(held), 'blocked': len(blocked), 'eval_files': [os.path.basename(f) for f in evfiles]}}
json.dump(rep, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
print(json.dumps({k: rep[k] for k in ('rows', 'extended', 'totals', 'max_tokens')}))
for k, v in rep['per_key'].items(): print(k, v)
print('TOKENS', json.dumps(lens)); print('FORMAT', json.dumps(fmt, indent=0))
