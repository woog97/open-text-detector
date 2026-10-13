"""Space-side T2.1 mix builder (T2 minus the T2.1 held-out split; GPT 840->600, Claude 4200->4440 rows/epoch) (runs in /tmp/pangram-t2-20261006). Writes runs/qwen35-4b-t21/prepared-v2 and build-t21.json.

Per stage-2 epoch at full length (24,000 rows; the trainer takes the first 20% of each dataset key in file order, so each
key's first 20% is filled with rows not used in earlier epochs whenever supply allows). Stage 1, selection and calibration
are copied unchanged. Every row must fit 510 tokens, pass never_train_hit (all held-out lists) and the
evaluation/selection/calibration leakage check (paper keys and shared sentences of 10+ words)."""
import gzip, hashlib, json, random, re, shutil, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

R = Path('/tmp/pangram-t2-20261006'); IN = R / 'inputs'; log = open(R / 'build-t21.log', 'a')
def say(**k): log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()

PLAN = {  # key: (full-length rows per epoch, source)
    'edit_claude': (4440, 'pool'), 'edit_luna': (3360, 'pool'), 'edit_gpt': (600, 'pool'),
    'human_paper_orig': (3000, 'prep:papers/human_original'), 'human_section': (1000, 'pool'), 'human_para_e': (1000, 'pool'), 'human_host': (1000, 'pool'),
    'para_luna': (1740, 'prep:papers/paragraph_generate'), 'para_e': (1160, 'pool'),
    'sections_ai': (1400, 'pool'),
    'mirrors': (1400, 'prep:mirrors'), 'fullpapers': (500, 'prep:fullpapers'), 'claude_fp': (500, 'pool'),
    'human': (1700, 'prep:human'), 'gradtex': (1200, 'prep:gradtex')}
GROUP = {'small edits': ['edit_claude', 'edit_luna', 'edit_gpt'], 'human paper text': ['human_paper_orig', 'human_section', 'human_para_e', 'human_host'],
         'paragraph edits': ['para_luna', 'para_e'], 'sections': ['sections_ai'], 'fully AI': ['mirrors', 'fullpapers', 'claude_fp'],
         'generic human': ['human'], 'GRADTEX': ['gradtex']}
assert sum(n for n, _ in PLAN.values()) == 24000

try:
    say(event='start')
    src = R / 'nt21.py'; ns = {}
    exec(src.read_text(), ns); never_train_hit = ns['never_train_hit']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R / 'assets/qwen35-4b'), local_files_only=True)
    base = R / 'runs/qwen35-4b/prepared-v2'; man = json.loads((base / 'manifest.json').read_text())
    SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
    def sents(t): return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}
    keys, held = set(), set()
    for r in (json.loads(l) for l in gzip.open(R / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt')):
        keys.add(r['id'].split('/')[0]); keys.add(str(r.get('group_id') or '').replace('paper:', '')); held |= sents(r['text'])
    for name in ['selection-windows', 'calibration-windows']:
        for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
            r = json.loads(l); keys.add(str(r.get('paper_id', '')).replace('paper:', '')); keys.add(r['id'].split('/')[0]); held |= sents(r['text'])
    keys.discard('')
    rejects = defaultdict(Counter)

    def crop510(r, limit=500):
        """Rows longer than the window: keep a <=limit-token window around the AI span (or target), snapped to whitespace."""
        t = r['text']; off = tok(t, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        if len(off) <= 510: return r
        ai = [g for g in r['regions'] if g['label'] == 1]
        a0 = min(g['start'] for g in ai) if ai else int(r.get('target_start', 0)); a1 = max(g['end'] for g in ai) if ai else int(r.get('target_end', len(t)))
        if not ai and a1 - a0 >= len(t) * .9: a0 = a1 = rng_crop.randrange(len(t))  # whole-row target: random window
        i0 = next((i for i, (a, b) in enumerate(off) if b > a0), 0); i1 = next((i for i, (a, b) in enumerate(off) if b >= a1), len(off) - 1)
        span = i1 - i0 + 1
        st = i0 if span >= limit else max(0, min(len(off) - limit, i0 - rng_crop.randrange(limit - span + 1)))
        cs, ce = off[st][0], off[min(len(off), st + limit) - 1][1]
        while cs > 0 and cs < len(t) and not t[cs - 1].isspace(): cs += 1
        while ce < len(t) and ce > cs and not t[ce - 1].isspace() and not t[ce].isspace(): ce -= 1
        txt = t[cs:ce]
        rg = [{'start': max(g['start'], cs) - cs, 'end': min(g['end'], ce) - cs, 'label': g['label']} for g in r['regions'] if min(g['end'], ce) > max(g['start'], cs)]
        cropped['n'] += 1
        return {**r, 'id': f"{r['id']}@{cs}", 'text': txt, 'regions': rg, 'source_start': cs, 'source_end': ce,
                'target_start': max(0, min(len(txt), int(r.get('target_start', 0)) - cs)), 'target_end': max(0, min(len(txt), int(r.get('target_end', len(t))) - cs))}

    rng_crop = random.Random(7); cropped = Counter()

    def screen(rows, cat):
        out = []
        rows = [crop510(r) if r.get('supervision') != 'document_only' else r for r in rows]
        n = [len(x) for x in tok([r['text'] for r in rows], add_special_tokens=False)['input_ids']] if rows else []
        for r, k in zip(rows, n):
            if not 0 < k <= 510: rejects[cat]['over_510_tokens'] += 1; continue
            if r.get('supervision') != 'document_only' and never_train_hit(r): rejects[cat]['never_train'] += 1; continue
            if r.get('supervision') == 'document_only' and never_train_hit({k2: v for k2, v in r.items() if k2 != 'regions'}): rejects[cat]['never_train'] += 1; continue
            ids = {str(r.get('paper_id', '')).replace('paper:', ''), r['id'].split('/')[0], str(r.get('group', '')).replace('paper:', '').replace('human:', '')}
            if ids & keys: rejects[cat]['eval_paper_id'] += 1; continue
            if sents(r['text']) & held: rejects[cat]['eval_shared_sentence'] += 1; continue
            out.append(r)
        return out

    pools = defaultdict(list)
    for l in gzip.open(IN / 't2-pools.jsonl.gz', 'rt'):
        r = json.loads(l); pools[r.pop('pool')].append(r)
    for k in list(pools): pools[k] = screen(pools[k], k)
    say(event='pools_screened', cropped=cropped['n'], sizes={k: len(v) for k, v in pools.items()}, rejects={k: dict(v) for k, v in rejects.items()})
    prep = []
    for e in range(3):
        rows = [json.loads(l) for l in gzip.open(base / f'stage2-epoch{e}.jsonl.gz', 'rt')]
        by = defaultdict(list)
        for r in rows:
            d = r['dataset']
            if d == 'papers': d = 'papers/' + r['id'].rsplit('/', 1)[-1]
            by[d].append(r)
        prep.append(by)
    seen_ids = {}
    for e in range(3):
        for d, rs in prep[e].items():
            prep[e][d] = screen(rs, 'prep:' + d)
    say(event='prepared_screened', rejects={k: dict(v) for k, v in rejects.items() if k.startswith('prep:')},
        per_epoch={e: {d: len(v) for d, v in prep[e].items()} for e in range(3)})

    rng = random.Random(20261006); used = defaultdict(set); report = {}
    out_dir = R / 'runs/qwen35-4b-t21/prepared-v2'
    if out_dir.exists(): shutil.rmtree(out_dir)
    shutil.copytree(base, out_dir); m2 = json.loads(json.dumps(man))
    for e in range(3):
        key = f'stage2-epoch{e}'; rows_out = []; rep = {}
        for cat, (n, source) in PLAN.items():
            if source == 'pool':
                cand = pools[cat]
            else:
                d = source.split(':', 1)[1]; cand = prep[e][d] + [r for j in range(3) if j != e for r in prep[j][d]]
            k20 = round(n * .2)
            fresh = [r for r in cand if r['id'] not in used[cat]]
            uniq_fresh = list({r['id']: r for r in fresh}.values())
            first = uniq_fresh[:k20]
            short20 = k20 - len(first)
            if short20 > 0:
                first += [rng.choice(cand) for _ in range(short20)]
            used[cat] |= {r['id'] for r in first}
            rest_src = [r for r in uniq_fresh[k20:]] or list({r['id']: r for r in cand}.values())
            rest = rest_src[:n - k20]
            while len(rest) < n - k20:
                rest.append(rest_src[(len(rest)) % len(rest_src)])
            for i, r in enumerate(first + rest):
                rows_out.append({**r, 'dataset': cat, 'draw_id': f'{key}-{cat}-{i}'})
            rep[cat] = {'rows': n, 'at_20pct': k20, 'at_20pct_reused_from_earlier_epochs': max(0, short20),
                        'full_length_distinct': len({r['id'] for r in first + rest}), 'supply_after_screening': len({r['id'] for r in cand})}
        b = ''.join(json.dumps(r) + '\n' for r in rows_out).encode()
        (out_dir / f'{key}.jsonl.gz').write_bytes(gzip.compress(b))
        m2['files'][key] = {**m2['files'][key], 'rows': len(rows_out), 'sha256': hashlib.sha256(b).hexdigest()}
        m2['counts'][key] = dict(Counter(r['dataset'] for r in rows_out)); report[key] = rep
    (out_dir / 'manifest.json').write_text(json.dumps(m2, indent=1))
    models = json.loads((R / 'models.json').read_text()); spec = next(m for m in models if m['name'] == 'qwen35-4b')
    if not any(m['name'] == 'qwen35-4b-t21' for m in models): models.append({**spec, 'name': 'qwen35-4b-t21'})
    (R / 'models.json').write_text(json.dumps(models, indent=1))
    if not (R / 'assets/qwen35-4b-t21').exists(): (R / 'assets/qwen35-4b-t21').symlink_to(R / 'assets/qwen35-4b')
    grouped = {key: {g: sum(report[key][c]['rows'] for c in cs) for g, cs in GROUP.items()} for key in report}
    res = {'per_epoch': report, 'grouped': grouped, 'rejects': {k: dict(v) for k, v in rejects.items()},
           'manifest_sha256': hashlib.sha256((out_dir / 'manifest.json').read_bytes()).hexdigest()}
    (R / 'build-t21.json').write_text(json.dumps(res, indent=1)); say(event='done')
except Exception as ex:
    say(event='failed', error=repr(ex), tb=traceback.format_exc()[-2000:])
