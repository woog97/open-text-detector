"""Omission-step eval: prepare judge batches from page kits, then score judge outputs.

The omission step (backend/open_text_detector_backend/omissions.py) runs on the training Space and
writes page kits: per page a JPEG and JSON with baseline_text and omission_text. This
script turns kits into cases for a fixed Sonnet judge (judge_prompt.md, run as Claude
Code subagents), validates the judge's files and writes results.jsonl / traces/ in the
build-eval layout under .claude/hillclimb/<flow>/<variant>/.

  python omission_eval.py prepare --variant baseline --kit DIR [--kit DIR ...]
  python omission_eval.py prepare --flow omissions-calibration --variant v1 --control sabotage --kit DIR
  python omission_eval.py score --variant baseline

Page images and texts are third-party paper content: the flow directory is gitignored.
"""
import argparse, difflib, json, math, random, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BATCH = 10
MARKER = re.compile(r"⟦[^⟧]*⟧")
# Rates are per 1,000 baseline words. Guardrail page flags are 0/1 floats so the
# report's headline (first binary metric, else first metric) is the loss rate.
METRICS = [
    {'id': 'prose_loss_per_1k', 'label': 'prose loss/1k', 'kind': 'float', 'scale': 100, 'better': 'lower'},
    {'id': 'math_loss_per_1k', 'label': 'math loss/1k', 'kind': 'float', 'scale': 100, 'better': 'lower'},
    {'id': 'removal_precision', 'label': 'removal prec', 'kind': 'float', 'scale': 1},
    {'id': 'noise_left_per_1k', 'label': 'noise left /1k', 'kind': 'float', 'scale': 1000, 'better': 'lower'},
    {'id': 'baseline_noise_per_1k', 'label': 'noise before', 'kind': 'float', 'scale': 1000, 'better': 'lower'},
    {'id': 'no_major_loss', 'label': 'no major loss', 'kind': 'float', 'scale': 1},
    {'id': 'noise_free', 'label': 'noise free', 'kind': 'float', 'scale': 1},
    {'id': 'clean_page', 'label': 'clean page', 'kind': 'float', 'scale': 1},
]
LOSS_TYPES = {'prose', 'heading', 'caption', 'list', 'footnote', 'theorem', 'algorithm', 'box', 'number', 'inline_math'}
NOISE_TYPES = {'figure_text', 'table_cells', 'display_math', 'garbled_math', 'stray_numbers', 'references', 'other'}


def case_id(name):
    return re.sub(r'^(worst|edge|standard|heldout)-', '', name)


def removed_runs(baseline, omission):
    """Baseline word runs absent from the omission text, with the marker that replaced each."""
    a = baseline.split(); b = omission.split()
    runs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ('delete', 'replace'):
            repl = ' '.join(b[j1:j2])
            runs.append({'i': len(runs), 'text': ' '.join(a[i1:i2]), 'replaced_by': ' '.join(MARKER.findall(repl)) if op == 'replace' else ''})
    return runs


def sabotage(text, seed):
    """Control: delete one ordinary prose sentence (the judge must flag it)."""
    sents = [m for m in re.finditer(r"[A-Z][^.!?\n]{40,300}[.!?]", text) if len(re.findall(r"\b[a-z]{3,}\b", m.group())) >= 8]
    if not sents: return text, None
    m = random.Random(seed).choice(sents)
    return text[:m.start()] + text[m.end():], m.group()


def prepare(a):
    flow = ROOT / '.claude/hillclimb' / a.flow; vdir = flow / a.variant
    (flow / 'inputs').mkdir(parents=True, exist_ok=True); (vdir / 'batches').mkdir(parents=True, exist_ok=True)
    state_path = flow / '_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'metrics': METRICS, 'perf_fields': [
        {'id': 'removed_words', 'label': 'removed words'}, {'id': 'baseline_words', 'label': 'baseline words'}]}
    cases = []
    for kit, split in zip(a.kit, a.split):
        manifest = json.loads((Path(kit) / 'manifest.json').read_text())
        for m in manifest:
            rec = json.loads((Path(kit) / f"{m['name']}.json").read_text())
            cid = case_id(m['name']); img = flow / 'inputs' / f'{cid}.jpg'
            if not img.exists(): shutil.copyfile(Path(kit) / f"{m['name']}.jpg", img)
            base = rec['baseline_text']; om = rec['omission_text']; meta = {'kit': Path(kit).name, 'page': rec['page']}
            if a.control == 'identity': om = base
            elif a.control == 'sabotage':
                om, cut = sabotage(base, cid)
                if cut is None: continue
                meta['sabotaged_sentence'] = cut
            cases.append({'id': cid, 'split': split, 'image': str(img), 'baseline_text': base, 'omission_text': om,
                          'removed': removed_runs(base, om), 'meta': meta, 'tags': [split, m.get('group', 'heldout')]})
            state.setdefault(f'{split}_ids', [])
            if cid not in state[f'{split}_ids']: state[f'{split}_ids'].append(cid)
    state_path.write_text(json.dumps(state, indent=1))
    (vdir / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    batches = [cases[i:i + BATCH] for i in range(0, len(cases), BATCH)]
    names = []
    for k, b in enumerate(batches):
        name = f'b{k:03d}'; names.append(name)
        (vdir / 'batches' / f'{name}.json').write_text(json.dumps(
            [{x: c[x] for x in ('id', 'image', 'baseline_text', 'omission_text', 'removed')} for c in b], ensure_ascii=False, indent=1))
    print(f'{len(cases)} cases in {len(names)} batches -> {vdir / "batches"}')


def wilson(k, n, z=1.96):
    if not n: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def check(j, case):
    """Fail loud on a malformed judgment rather than scoring it."""
    errs = []; cid = case['id']
    if j.get('image_viewed') is not True: errs.append('image not viewed')
    labels = j.get('removed_labels')
    if not isinstance(labels, list) or sorted(x.get('i', -1) for x in labels if isinstance(x, dict)) != [r['i'] for r in case['removed']]:
        errs.append('removed_labels do not cover every removed entry exactly once')
    else:
        for x in labels:
            if x.get('label') not in ('noise', 'author', 'mixed'): errs.append(f'bad label {x!r:.80}')
            elif x['label'] != 'noise' and (x.get('type') not in LOSS_TYPES or x.get('severity') not in ('low', 'medium', 'high')):
                errs.append(f'author label without type/severity {x!r:.80}')
            elif x['label'] == 'mixed' and not str(x.get('author_words', '')).strip(): errs.append('mixed without author_words')
    if not isinstance(j.get('noise_remaining'), list): errs.append('noise_remaining missing')
    else:
        for x in j['noise_remaining']:
            if not isinstance(x, dict) or not x.get('quote') or x.get('type') not in NOISE_TYPES: errs.append(f'bad noise item {x!r:.80}')
    if j.get('baseline_noise') not in ('none', 'some', 'heavy'): errs.append('bad baseline_noise')
    if not isinstance(j.get('marker_errors'), list): errs.append('marker_errors missing')
    return [f'{cid}: {e}' for e in errs]


def score(a):
    flow = ROOT / '.claude/hillclimb' / a.flow; vdir = flow / a.variant
    cases = {c['id']: c for c in json.loads((vdir / 'cases.json').read_text())}
    judged = {}; errors = []
    for f in sorted((vdir / 'judgments').glob('*.json')):
        for j in json.loads(f.read_text()):
            if j.get('id') not in cases: errors.append(f'{f.name}: unknown id {j.get("id")}'); continue
            e = check(j, cases[j['id']])
            if e: errors += e
            else: judged[j['id']] = j
    missing = [c for c in cases if c not in judged]
    if errors:
        print('\n'.join(errors[:30])); sys.exit(f'{len(errors)} malformed judgments; nothing scored')
    if missing and not a.partial: sys.exit(f'{len(missing)} cases not judged yet (first: {missing[:5]}); use --partial to score anyway')
    (vdir / 'traces').mkdir(exist_ok=True)
    rows = []; agg = {}
    for cid, j in judged.items():
        c = cases[cid]; noise = j['noise_remaining']; removed = {r['i']: r for r in c['removed']}
        words = max(1, len(c['baseline_text'].split()))
        lost = []; noise_removed = 0
        for x in j['removed_labels']:
            n = len(removed[x['i']]['text'].split())
            if x['label'] == 'noise': noise_removed += n; continue
            k = n if x['label'] == 'author' else min(n, len(str(x['author_words']).split()))
            noise_removed += n - k
            lost.append({'words': k, 'severity': x['severity'], 'type': x['type'], 'text': removed[x['i']]['text'] if x['label'] == 'author' else x['author_words']})
        lost_words = sum(x['words'] for x in lost); removed_words = sum(len(r['text'].split()) for r in c['removed'])
        noise_left = sum(len(x['quote'].split()) for x in noise)
        prose_lost = sum(x['words'] for x in lost if x['type'] != 'inline_math')
        grade = {'prose_loss_per_1k': 1000 * prose_lost / words, 'math_loss_per_1k': 1000 * (lost_words - prose_lost) / words, 'noise_left_per_1k': 1000 * noise_left / words,
                 'baseline_noise_per_1k': 1000 * (noise_removed + noise_left) / words,
                 'no_major_loss': float(not any(x['severity'] != 'low' for x in lost)),
                 'noise_free': float(not noise), 'clean_page': float(not lost and not noise)}
        if removed_words: grade['removal_precision'] = noise_removed / removed_words
        expl = {'prose_loss_per_1k': '; '.join(f"[{x['severity']} {x['type']}] {x['text'][:120]}" for x in lost) or 'none lost',
                'noise_left_per_1k': '; '.join(f"[{x['type']}] {x['quote'][:80]}" for x in noise) or 'no noise left'}
        meta = dict(c['meta'], baseline_noise=j['baseline_noise'], marker_errors=len(j['marker_errors']))
        if 'sabotaged_sentence' in meta:   # control: did the judge catch the deleted sentence?
            cut = set(re.findall(r'[a-z]{4,}', meta['sabotaged_sentence'].lower()))
            meta['sabotage_caught'] = any(len(cut & set(re.findall(r'[a-z]{4,}', x['text'].lower()))) >= .5 * len(cut) for x in lost)
        rows.append({'prompt_id': cid, 'prompt': c['omission_text'], 'tags': c['tags'], 'status': 'ok', 'model': 'claude-sonnet-5-5',
                     'grade': grade, 'explanation': expl, 'meta': meta,
                     'removed_words': sum(len(r['text'].split()) for r in c['removed']), 'baseline_words': len(c['baseline_text'].split()),
                     'attachments': [{'kind': 'image', 'ref': f'inputs/{cid}.jpg', 'alt': 'page image'}]})
        (vdir / 'traces' / f'{cid}_rep0.json').write_text(json.dumps([
            {'role': 'user', 'content': 'BASELINE TEXT\n\n' + c['baseline_text']},
            {'role': 'assistant', 'content': 'OMISSION TEXT\n\n' + c['omission_text']},
            {'role': 'tool_result', 'name': 'judge', 'content': json.dumps(j, ensure_ascii=False, indent=1)}], ensure_ascii=False))
        for m, v in grade.items(): agg.setdefault((c['split'], m), []).append(v)
    with open(vdir / 'results.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{a.flow}/{a.variant}: {len(rows)} judged' + (f', {len(missing)} missing' if missing else ''))
    for m in [x['id'] for x in METRICS]:
        for split in sorted({s for s, _ in agg}):
            vals = agg.get((split, m))
            if not vals: continue
            if m in ('no_major_loss', 'noise_free', 'clean_page'):
                p, lo, hi = wilson(int(sum(vals)), len(vals)); print(f'  {split:6s} {m:22s} {p:7.1%}  [{lo:.1%}, {hi:.1%}]  n={len(vals)}')
            else:
                mu = sum(vals) / len(vals); sd = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** .5
                med = sorted(vals)[len(vals) // 2]
                print(f'  {split:6s} {m:22s} mean {mu:7.2f} ±{1.96 * sd / len(vals) ** .5:.2f}  median {med:.2f}  n={len(vals)}')
    caught = [r['meta']['sabotage_caught'] for r in rows if 'sabotage_caught' in r['meta']]
    if caught: print(f'  sabotage caught {sum(caught)}/{len(caught)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--flow', default='omissions'); q.add_argument('--variant', required=True)
    q.add_argument('--kit', action='append', required=True); q.add_argument('--split', action='append', required=True)
    q.add_argument('--control', choices=['identity', 'sabotage'])
    s = sub.add_parser('score'); s.add_argument('--flow', default='omissions'); s.add_argument('--variant', required=True)
    s.add_argument('--partial', action='store_true')
    a = p.parse_args()
    if a.cmd == 'prepare' and len(a.kit) != len(a.split): sys.exit('give one --split per --kit')
    {'prepare': prepare, 'score': score}[a.cmd](a)
