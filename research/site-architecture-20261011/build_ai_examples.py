"""Fixed sample of AI-text training rows for the Data page (AI text tab).

Reads the shipped 72,000-row mix and keeps up to PER rows for every (key, writer, operation), chosen with a fixed seed,
plus the matching human row of the same passage when the mix has one (paragraph edits, Luna paragraphs, sections).
The output holds paper text, so it is uploaded to R2 (indexes/transparency/ai-examples.json), never committed.

Usage (on the H200): python3 build_ai_examples.py MIX.jsonl.gz OUT.json
"""
import gzip, json, random, re, sys
from collections import defaultdict

PER = 3; SEED = 20261011
KIND = {'edit_claude': 'small', 'edit_luna': 'small', 'edit_deepseek': 'small', 'edit_gpt': 'small',
        'para_e': 'para', 'para_deepseek': 'para', 'para_luna': 'para', 'sections_ai': 'section',
        'mirrors': 'full', 'fullpapers': 'full', 'claude_fp': 'full'}
WRITER = {'claude-opus-5-5': 'Claude Opus 5.5', 'claude-sonnet-5-5': 'Claude Sonnet 5.5', 'claude-haiku-4-5': 'Claude Haiku 4.5',
          'openai/gpt-6-luna': 'GPT-6 Luna', 'gpt-6-luna': 'GPT-6 Luna', 'openai/gpt-6.1-sol': 'GPT-6.1 Sol', 'gpt-6.1-sol': 'GPT-6.1 Sol',
          'deepseek/deepseek-v4.1-flash': 'DeepSeek V4.1 Flash'}
OPS = ('rewrite_one', 'rewrite_two', 'insert_one', 'split_one', 'merge_two', 'draft', 'rewrite')
SOURCE = [(r'^arxiv:', 'arXiv'), (r'^pmc:', 'PubMed Central'), (r'^pes2o:', 'peS2o'), (r'^acl:', 'ACL Anthology'), (r'^or:', 'OpenReview'),
          (r'^(icml|neurips|acl|nips)-\d{4}', 'ML conference paper')]


def operation(key, rid):
    last = rid.split('@')[0].rsplit('/', 1)[-1]
    if last.startswith('splice-one'): return 'splice_one'
    if last.startswith('splice-two'): return 'splice_two'
    if last == 'paragraph_generate': return 'draft_from_notes'
    for op in OPS:
        if last.endswith(op): return op
    return {'mirrors': 'mirror', 'fullpapers': 'manuscript', 'claude_fp': 'claude_paper'}.get(key, 'section')


def source(rid, fmt):
    for pat, name in SOURCE:
        if re.match(pat, rid): return name
    return {'paired_pdf_v1': 'ML conference paper', 'clean_v2': 'OpenReview', 'generated_doc': None, 'generated_markdown': None}.get(fmt)


def regions(r):
    return [list(x) if isinstance(x, (list, tuple)) else [x['start'], x['end'], x['label']] for x in r.get('regions') or []]


def human_key(rid):
    """id prefix shared by an AI row and its matched human row (crop offsets dropped)."""
    base = rid.split('@')[0]
    if base.endswith('/human') or base.endswith('/human_original'): return base.rsplit('/', 1)[0]
    if re.search(r'/([\w.-]+-)?(draft|rewrite)$|/paragraph_generate$|/splice-(one|two)-\d+$', base): return base.rsplit('/', 1)[0]
    return None


def main(mix, out):
    rows = [json.loads(l) for l in gzip.open(mix, 'rt')]
    groups = defaultdict(list); humans = {}
    for i, r in enumerate(rows):
        k = r['dataset']
        if k in KIND: groups[(k, r.get('generator') or ('openai/gpt-6-luna' if k == 'mirrors' else None), operation(k, r['id']))].append(i)
        elif k in ('human_para_e', 'human_section', 'human_paper_orig') and human_key(r['id']): humans.setdefault(human_key(r['id']), i)
    rng = random.Random(SEED); picked = []
    for g in sorted(groups, key=lambda g: (g[0], str(g[1]), g[2])):
        idx = sorted(groups[g]); rng.shuffle(idx)
        for i in idx[:PER]:
            r = rows[i]; hk = human_key(r['id']); h = humans.get(hk) if hk else None
            picked.append({'id': r['id'].split('@')[0], 'key': r['dataset'], 'kind': KIND[r['dataset']],
                           'writer': WRITER.get(g[1], g[1]), 'op': g[2], 'source': source(r['id'], r.get('text_format')),
                           'format': r.get('text_format'), 'text': r['text'], 'regions': regions(r), 'group_rows': len(groups[g]),
                           'human': {'text': rows[h]['text']} if h is not None else None})
    json.dump({'built': 'final-20261007 mix, stage2-epoch0 (72,000 rows)', 'seed': SEED, 'per_group': PER, 'examples': picked},
              open(out, 'w'), ensure_ascii=False, separators=(',', ':'))
    print(len(picked), 'examples from', len(groups), 'groups;', sum(1 for p in picked if p['human']), 'with a human version')


if __name__ == '__main__':
    main(*sys.argv[1:3])
