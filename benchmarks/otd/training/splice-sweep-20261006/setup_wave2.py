"""Wave 2 Space-side setup in /tmp/pangram-splice-20261006: LLE and MIX prepared variants, cross-model dataset, pyarrow."""
import gzip, hashlib, json, random, re, shutil, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

R = Path('/tmp/pangram-splice-20261006'); log = open(R / 'setup-wave2.log', 'a')


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()


try:
    say(event='start')
    try:
        import pyarrow  # noqa
        say(event='pyarrow', status='present', version=pyarrow.__version__)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', '/tmp/pangram-wandb-vendor', 'pyarrow'], check=True, capture_output=True)
        say(event='pyarrow', status='installed')
    from huggingface_hub import snapshot_download
    p = snapshot_download('open-text-detector/heterogeneous-ai-spans', repo_type='dataset', local_dir=str(R / 'hetero'),
                          allow_patterns=['data/mixed/*', 'data/human_controls/*', 'README.md', 'SHA256SUMS.txt'])
    say(event='dataset', files=sorted(str(x.relative_to(R / 'hetero')) for x in (R / 'hetero/data').rglob('*.parquet')))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R / 'assets/qwen35-4b'), local_files_only=True)
    base = R / 'runs/qwen35-4b/prepared-v2'; man = json.loads((base / 'manifest.json').read_text())
    SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)

    def sents(t):
        return {' '.join(w) for m in SENT.finditer(t) for w in [re.findall(r'[a-z0-9]+', m.group().lower())] if len(w) >= 10}

    keys, held = set(), set()
    for r in (json.loads(l) for l in gzip.open(R / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt')):
        keys.add(r['id'].split('/')[0]); keys.add(str(r.get('group_id') or '').replace('paper:', '')); held |= sents(r['text'])
    for name in ['selection-windows', 'calibration-windows']:
        for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
            r = json.loads(l); keys.add(str(r.get('paper_id', '')).replace('paper:', '')); keys.add(r['id'].split('/')[0]); held |= sents(r['text'])
    keys.discard('')
    train_sents = set()
    for e in range(3):
        for l in gzip.open(base / f'stage2-epoch{e}.jsonl.gz', 'rt'):
            r = json.loads(l)
            if r.get('regions'):
                train_sents |= sents(r['text'])

    def load_edits(path, kind, ds):
        raw = [json.loads(l) for l in gzip.open(path, 'rt')]
        for s in raw:
            if isinstance(s['regions'], str):
                import ast; s['regions'] = ast.literal_eval(s['regions'])
            s['regions'] = [{'start': int(r['start']), 'end': int(r['end']), 'label': int(r['label'])} for r in s['regions']]
            s['target_start'], s['target_end'] = int(s['target_start']), int(s['target_end'])
        n = [len(x) for x in tok([s['text'] for s in raw], add_special_tokens=False)['input_ids']]
        fit = [s for s, k in zip(raw, n) if 0 < k <= 510]
        by_id = by_text = 0; kept = []
        for s in fit:
            if {s['paper_id'], s['id'].split('/')[0], s['group'].replace('paper:', '')} & keys:
                by_id += 1; continue
            if sents(s['text']) & held:
                by_text += 1; continue
            kept.append(s)
        stats = {'in': len(raw), 'over_510_tokens': len(raw) - len(fit), 'removed_paper_id': by_id, 'removed_shared_sentence': by_text, 'kept': len(kept),
                 'kept_papers': len({s['group'] for s in kept}), 'by_size': dict(Counter(s['edit_size'] for s in kept)),
                 'kept_sharing_sentence_with_stage2_training': sum(bool(sents(s['text']) & train_sents) for s in kept)}
        if 'edit_type' in kept[0]:
            stats['by_type'] = dict(Counter(s['edit_type'] for s in kept))
        random.Random(20261006).shuffle(kept)
        rows = [{'id': s['id'], 'paper_id': s['paper_id'], 'kind': kind, 'text': s['text'], 'source_start': 0, 'source_end': len(s['text']),
                 'regions': s['regions'], 'target_start': s['target_start'], 'target_end': s['target_end'], 'dataset': ds, 'group': s['group'],
                 'edit_size': s['edit_size']} for s in kept]
        return rows, stats

    llm, st_llm = load_edits(R / 'llm-edits-v1.jsonl.gz', 'llm_edit', 'papers_llm_edit'); say(event='llm_edits', **st_llm)
    spl, st_spl = load_edits(R / 'splices-v1.jsonl.gz', 'splice', 'papers_splice'); say(event='splices', **st_spl)

    models = json.loads((R / 'models.json').read_text()); spec = next(m for m in models if m['name'] == 'qwen35-4b')

    def build(name, plan):
        out = R / 'runs' / name / 'prepared-v2'
        if out.exists(): shutil.rmtree(out)
        shutil.copytree(base, out); m2 = json.loads(json.dumps(man)); per = {}
        for e in range(3):
            key = f'stage2-epoch{e}'; rows = [json.loads(l) for l in gzip.open(base / f'{key}.jsonl.gz', 'rt')]
            slots = plan(rows)  # list of (row index, source list)
            used = Counter()
            for i, src in slots:
                pool = llm if src == 'llm' else spl
                j = e * (len(pool) // 3) + used[src]; used[src] += 1
                rows[i] = {**pool[j % len(pool)], 'draw_id': f'{key}-{src}-{used[src] - 1}'}
            b = ''.join(json.dumps(r) + '\n' for r in rows).encode()
            (out / f'{key}.jsonl.gz').write_bytes(gzip.compress(b))
            m2['files'][key] = {**m2['files'][key], 'rows': len(rows), 'sha256': hashlib.sha256(b).hexdigest()}
            m2['counts'][key] = dict(Counter(r['dataset'] for r in rows)); per[key] = m2['counts'][key]
            per[key + '@20%'] = {k: round(v * .2) for k, v in m2['counts'][key].items()}
            per[key + '_distinct_new_rows'] = {k: v for k, v in used.items()}
        (out / 'manifest.json').write_text(json.dumps(m2, indent=1))
        if not (R / 'assets' / name).exists():
            (R / 'assets' / name).symlink_to(R / 'assets/qwen35-4b')
        if not any(m['name'] == name for m in models):
            models.append({**spec, 'name': name})
        say(event='variant_built', name=name, per_epoch=per)

    build('qwen35-4b-lle', lambda rows: [(i, 'llm') for i, r in enumerate(rows) if r['dataset'] == 'gradtex'])

    def mix(rows):
        idx = [i for i, r in enumerate(rows) if r['dataset'] == 'mirrors'][1::2]
        return [(i, 'splice' if k % 2 == 0 else 'llm') for k, i in enumerate(idx)]
    build('qwen35-4b-mix', mix)
    (R / 'models.json').write_text(json.dumps(models, indent=1))
    say(event='done')
except Exception as ex:
    say(event='failed', error=repr(ex), tb=traceback.format_exc()[-1500:])
