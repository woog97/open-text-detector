"""Space-side setup for the splice experiment (runs detached in /tmp/pangram-splice-20261006).

Copies the qwen35-4b assets, vendor packages and prepared-v2 data, removes splices that leak into the evaluation,
selection or calibration data, and builds two prepared variants (stage 2 only; stage 1 unchanged):
  qwen35-4b-sph: half of each epoch's 'papers' rows (every second one in file order) replaced by splices
  qwen35-4b-spg: every 'gradtex' row replaced by splices (paper pairs kept intact)
Splice slices differ per epoch, so the 20%-length prefix of each epoch uses distinct splices.
"""
import ast, gzip, hashlib, json, random, re, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

R = Path('/tmp/pangram-splice-20261006'); log = open(R / 'setup.log', 'a')


def say(**k):
    log.write(json.dumps({'t': time.strftime('%H:%M:%S'), **k}) + '\n'); log.flush()


try:
    say(event='start')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--target', '/tmp/pangram-wandb-vendor', 'wandb'], check=True,
                   capture_output=True)
    say(event='wandb_installed')
    shutil.copytree('/tmp/pangram-space-fast10/vendor', R / 'vendor')
    src = Path('/tmp/pangram-space-fast10/assets/qwen35-4b'); ref = Path('/data/workspace/backbone-launch-20261003/assets/qwen35-4b')
    dst = R / 'assets/qwen35-4b'; shutil.copytree(src, dst)
    for p in ref.iterdir():
        if p.is_file():
            assert (dst / p.name).stat().st_size == p.stat().st_size, p.name
    say(event='assets_copied', files=len(list(dst.iterdir())))
    P0 = Path('/data/workspace/backbone-launch-20261003/runs/qwen35-4b/prepared-v2'); base = R / 'runs/qwen35-4b/prepared-v2'
    shutil.copytree(P0, base)
    man = json.loads((base / 'manifest.json').read_text())
    for k, v in man['files'].items():
        assert hashlib.sha256(gzip.decompress((base / f'{k}.jsonl.gz').read_bytes())).hexdigest() == v['sha256'], k
    say(event='prepared_copied', manifest_sha256=hashlib.sha256((base / 'manifest.json').read_bytes()).hexdigest(), files=sorted(man['files']))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(dst), local_files_only=True)
    sp = [json.loads(l) for l in gzip.open(R / 'splices-v1.jsonl.gz', 'rt')]
    for s in sp:
        if isinstance(s['regions'], str):
            s['regions'] = ast.literal_eval(s['regions'])
        s['regions'] = [{'start': int(r['start']), 'end': int(r['end']), 'label': int(r['label'])} for r in s['regions']]
        s['target_start'], s['target_end'] = int(s['target_start']), int(s['target_end'])
    n0 = len(sp)
    ntok = [len(x) for x in tok([s['text'] for s in sp], add_special_tokens=False)['input_ids']]
    sp = [s for s, n in zip(sp, ntok) if 0 < n <= 510]; n_tok = len(sp)

    # Leakage: paper keys and shared sentences (>= 10 words) with evaluation, selection and calibration data.
    SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)
    def sents(t):
        out = set()
        for m in SENT.finditer(t):
            w = re.findall(r'[a-z0-9]+', m.group().lower())
            if len(w) >= 10:
                out.add(' '.join(w))
        return out
    keys, held = set(), set()
    ev = [json.loads(l) for l in gzip.open(R / 'sweeps/sweep-eval-rows.jsonl.gz', 'rt')]
    for r in ev:
        keys.add(r['id'].split('/')[0]); keys.add(str(r.get('group_id') or '').replace('paper:', '')); held |= sents(r['text'])
    for name in ['selection-windows', 'calibration-windows']:
        for l in gzip.open(base / f'{name}.jsonl.gz', 'rt'):
            r = json.loads(l); keys.add(str(r.get('paper_id', '')).replace('paper:', '')); keys.add(r['id'].split('/')[0]); held |= sents(r['text'])
    keys.discard('')
    by_id = by_text = 0; kept = []
    for s in sp:
        k = {s['paper_id'], s['id'].split('/')[0], s['group'].replace('paper:', '')}
        if k & keys:
            by_id += 1; continue
        if sents(s['text']) & held:
            by_text += 1; continue
        kept.append(s)
    say(event='leakage', splices_in=n0, over_510_tokens=n0 - n_tok, removed_paper_id=by_id, removed_shared_sentence=by_text, kept=len(kept),
        kept_papers=len({s['group'] for s in kept}), by_size=dict(Counter(s['edit_size'] for s in kept)))
    random.Random(20261006).shuffle(kept)

    def clean(s, draw):
        return {'id': s['id'], 'paper_id': s['paper_id'], 'kind': 'splice', 'text': s['text'], 'source_start': 0, 'source_end': len(s['text']),
                'regions': s['regions'], 'target_start': s['target_start'], 'target_end': s['target_end'], 'dataset': 'papers_splice',
                'group': s['group'], 'draw_id': draw, 'edit_size': s['edit_size']}

    models = json.loads((R / 'models.json').read_text()); spec = next(m for m in models if m['name'] == 'qwen35-4b')
    for name, swap in [('qwen35-4b-sph', 'papers'), ('qwen35-4b-spg', 'gradtex')]:
        out = R / 'runs' / name / 'prepared-v2'; shutil.copytree(base, out); m2 = json.loads(json.dumps(man))
        for e in range(3):
            key = f'stage2-epoch{e}'; rows = [json.loads(l) for l in gzip.open(base / f'{key}.jsonl.gz', 'rt')]
            idx = [i for i, r in enumerate(rows) if r['dataset'] == swap]; drop = idx[1::2] if swap == 'papers' else idx
            start = e * (len(kept) // 3)
            for j, i in enumerate(drop):
                rows[i] = clean(kept[(start + j) % len(kept)], f'{key}-splice-{j}')
            b = ''.join(json.dumps(r) + '\n' for r in rows).encode()
            (out / f'{key}.jsonl.gz').write_bytes(gzip.compress(b))
            m2['files'][key] = {**m2['files'][key], 'rows': len(rows), 'sha256': hashlib.sha256(b).hexdigest()}
            m2['counts'][key] = dict(Counter(r['dataset'] for r in rows))
        (out / 'manifest.json').write_text(json.dumps(m2, indent=1))
        (R / 'assets' / name).symlink_to(dst)
        if not any(m['name'] == name for m in models):
            models.append({**spec, 'name': name})
        say(event='variant_built', name=name, counts=m2['counts']['stage2-epoch0'])
    (R / 'models.json').write_text(json.dumps(models, indent=1))
    say(event='done')
except Exception as ex:
    import traceback
    say(event='failed', error=repr(ex), tb=traceback.format_exc()[-1500:])
