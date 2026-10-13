"""Local half of the final single-pass mix: assemble the candidate pools from the ingested generation datasets on this Mac
(no tokenizer; mix_build.py on the Space crops, screens and mixes). Writes OUT/mix-inputs.tgz with
  pools.jsonl.gz        candidate rows, each tagged 'pool' (mix key), 'text_format' and 'generator'
  never-train/*.json    every held-out list in build_llm_edits.NEVER_TRAIN_FILES (the Space adds wide1's list)
  local-stats.json      raw pool sizes by generator and text format

  python mix_pools.py OUT_DIR
"""
import ast, collections, gzip, io, json, random, re, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]; P = ROOT / 'research'
sys.path.insert(0, str(P / 'llm-sentence-edits-20261005'))
EXCLUDED_GEN = re.compile(r'fable|gpt-6-sol|glm|mimo|space-bunny', re.I)  # held-out or eval-only writers: never trained on
rng = random.Random(20261007)


def rd(f):
    f = Path(f)
    if not f.exists(): return []
    return [json.loads(l) for l in (gzip.open(f, 'rt') if f.suffix == '.gz' else open(f))]


def regs(r):
    x = r['regions']; x = ast.literal_eval(x) if isinstance(x, str) else x
    return [{'start': int(g['start']), 'end': int(g['end']), 'label': int(g['label'])} for g in x]


def fmt_host(source):
    """Text format of a hosted row: wide-pool hosts ('openreview', host ids 'or:') are corpus clean v2; the older hosts
    come from PMC JATS, peS2o and arXiv text."""
    return 'clean_v2' if source in ('openreview', 'corpus') else f'host_{source}'


def row(r, pool, kind, fmt):
    t = r['text']
    return {'id': r['id'], 'paper_id': str(r.get('paper_id', r['id'])), 'group': str(r.get('group') or r.get('paper_id') or r['id']), 'kind': kind,
            'text': t, 'source_start': 0, 'source_end': len(t), 'regions': regs(r),
            'target_start': int(r.get('target_start', 0)), 'target_end': int(r.get('target_end', len(t))),
            'pool': pool, 'generator': r.get('generator') or r.get('model'), 'text_format': fmt, 'host_id': r.get('host_id')}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); pools = collections.defaultdict(list)
    # Claude small edits: paired-paper rows (claude-v1) + hosted rows; Haiku at most every 10th row of the ordered pool.
    hsm = rd(P / 'claude-hosted-edits-20261006/llm-edits-claude-hsm-v1.jsonl.gz')
    cl = [row(r, 'edit_claude', 'llm_edit', 'paired_pdf_v1') for r in rd(P / 'llm-sentence-edits-20261005/llm-edits-claude-v1.jsonl.gz')]
    cl += [row(r, 'edit_claude', 'llm_edit', fmt_host(r.get('source'))) for r in hsm
           if str(r['generator']).startswith('claude') and not EXCLUDED_GEN.search(str(r['generator']))]
    big = [r for r in cl if 'haiku' not in r['generator']]; small = [r for r in cl if 'haiku' in r['generator']]
    rng.shuffle(big); rng.shuffle(small)
    while big or small:
        for _ in range(9):
            if big: pools['edit_claude'].append(big.pop())
        if small: pools['edit_claude'].append(small.pop())
    # GPT small edits: GPT-6.1 Sol only (GPT-6 Sol is a held-out writer). Codex round-3 rows land in the same file when ingested.
    gp = [row(r, 'edit_gpt', 'llm_edit', fmt_host(r.get('source'))) for r in hsm if r['generator'] == 'openai/gpt-6.1-sol']
    rng.shuffle(gp); pools['edit_gpt'] = gp
    lu = rd(P / 'splice-edits-20261005/splices-v1.jsonl.gz') + rd(P / 'llm-sentence-edits-20261005/llm-edits-v1.jsonl.gz') + rd(P / 'llm-sentence-edits-20261005/llm-edits-pilot.jsonl.gz')
    for r in lu: r.setdefault('generator', 'openai/gpt-6-luna')
    rng.shuffle(lu); pools['edit_luna'] = [row(r, 'edit_luna', 'splice' if 'splice' in r['id'] else 'llm_edit', 'paired_pdf_v1') for r in lu]
    # Sections (AI and matched human); drop every group that has a Fable section, human side included.
    sec = rd(P / 'claude-sections-20261006/claude-sections-v1-train.jsonl.gz')
    bad = {r.get('group') for r in sec if EXCLUDED_GEN.search(str(r.get('generator')))}
    sec = [r for r in sec if r.get('group') not in bad]; rng.shuffle(sec)
    sfmt = lambda r: {'paired': 'paired_pdf_v1', 'archive': 'archive_v1', 'corpus': 'clean_v2'}.get(r.get('source'), str(r.get('source')))
    pools['sections_ai'] = [row(r, 'sections_ai', 'section', sfmt(r)) for r in sec if r['dataset'] == 'papers_section']
    pools['human_section'] = [row(r, 'human_section', 'human_matched', sfmt(r)) for r in sec if r['dataset'] == 'papers_section_human']
    # Claude/GPT-6.1 paragraph edits and their matched human paragraphs.
    pe = rd(P / 'claude-hosted-edits-20261006/paragraph-edits-claude-hsm-v1-train.jsonl.gz')
    drop = {r['id'] for r in pe if r['dataset'] == 'hsm_paragraph_edit' and EXCLUDED_GEN.search(str(r['generator']))}
    pe = [r for r in pe if r['id'] not in drop and r.get('matched_id') not in drop]; rng.shuffle(pe)
    pools['para_e'] = [row(r, 'para_e', 'paragraph_edit', fmt_host(r.get('source'))) for r in pe if r['dataset'] == 'hsm_paragraph_edit']
    pools['human_para_e'] = [row(r, 'human_para_e', 'human_matched', fmt_host(r.get('source'))) for r in pe if r['dataset'] == 'hsm_paragraph_edit_human']
    # Older human hosts (PMC/peS2o/arXiv) never assigned to any writer, in any items file (run or not).
    H = P / 'claude-hosted-edits-20261006'
    used = {r['host_id'] for r in hsm} | {r['host_id'] for r in pe if r.get('host_id')}
    for f in H.glob('items-*.jsonl'):
        used |= {json.loads(l)['host_id'] for l in open(f)}
    hosts = [h for f in ('hosts-raw.jsonl', 'hosts-gap-raw.jsonl') for h in rd(H / f) if h['host_id'] not in used]
    rng.shuffle(hosts)
    for h in hosts:
        t = '\n\n'.join(x for x in (h['prev'], h['target'], h['next']) if x)
        pools['human_host'].append({'id': 'host:' + h['host_id'], 'paper_id': h['record_id'], 'group': 'human:' + h['record_id'], 'kind': 'human_host', 'text': t,
                                    'source_start': 0, 'source_end': len(t), 'regions': [{'start': 0, 'end': len(t), 'label': 0}], 'target_start': 0,
                                    'target_end': len(t), 'pool': 'human_host', 'generator': None, 'text_format': fmt_host(h['source']), 'host_id': h['host_id']})
    # Claude full papers: ~1,600-character windows at whitespace, regions cropped.
    wins = []
    for r in rd(P / 'claude-fullpapers-20261006/claude-fullpapers-v1-train.jsonl.gz'):
        if EXCLUDED_GEN.search(str(r.get('model') or r.get('generator'))): continue
        t = r['text']; rg = regs(r); a = 0
        while a < len(t) - 200:
            b = min(len(t), a + 1600)
            while b < len(t) and not t[b - 1].isspace(): b -= 1
            txt = t[a:b]; cr = [{'start': max(g['start'], a) - a, 'end': min(g['end'], b) - a, 'label': g['label']} for g in rg if min(g['end'], b) > max(g['start'], a)]
            wins.append({'id': f"{r['id']}/w{a}", 'paper_id': str(r['paper_id']), 'group': str(r['group']), 'kind': 'claude_fullpaper_window', 'text': txt,
                         'source_start': a, 'source_end': b, 'regions': cr, 'target_start': 0, 'target_end': len(txt), 'pool': 'claude_fp',
                         'generator': r.get('model'), 'seed_id': r.get('seed_id'), 'text_format': 'generated_markdown'})
            a = b
    rng.shuffle(wins); pools['claude_fp'] = wins
    for k, v in pools.items():
        assert not any(EXCLUDED_GEN.search(str(r.get('generator'))) for r in v), k
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as g:
        for v in pools.values():
            for r in v: g.write((json.dumps(r, ensure_ascii=False) + '\n').encode())
    import build_llm_edits as ble
    stats = {k: {'rows': len(v), 'by_generator': dict(collections.Counter(str(r['generator']) for r in v)),
                 'by_format': dict(collections.Counter(r['text_format'] for r in v))} for k, v in pools.items()}
    with tarfile.open(out / 'mix-inputs.tgz', 'w:gz') as t:
        def add(name, b):
            ti = tarfile.TarInfo(name); ti.size = len(b); t.addfile(ti, io.BytesIO(b))
        add('pools.jsonl.gz', buf.getvalue())
        for f in ble.NEVER_TRAIN_FILES:
            if f.exists(): add('never-train/' + str(f.relative_to(ROOT)).replace('/', '__'), f.read_bytes())
        add('local-stats.json', json.dumps(stats, indent=1).encode())
    print(json.dumps({k: v['rows'] for k, v in stats.items()}))


if __name__ == '__main__':
    main(sys.argv[1])
