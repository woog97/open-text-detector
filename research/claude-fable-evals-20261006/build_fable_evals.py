"""Fable 5.1 evaluation-only sets (never trained on): light polish, humanizer, simulated co-writing.

Sources are held out already:
  polish, cowrite - Job E held-out host passages (human-source-mix pmc/pes2o/arxiv, <=2022), whose record ids and
                    paragraphs are on research/claude-hosted-edits-20261006/heldout-never-train.json. Never written by
                    any job (Job E was not run).
  humanizer       - AI paragraphs from the held-out Claude sections (claude-sections-v1-heldout.jsonl.gz), with the
                    section items' matched human rows as controls.
Generated text is added to heldout-never-train.json in this folder (loaded by build_llm_edits.never_train_hit).

Usage:
  build_fable_evals.py prepare
  build_fable_evals.py ingest {polish,humanizer,cowrite}       # works on partial outputs
"""
import argparse, os, difflib, gzip, hashlib, json, random, re, sys, time
from collections import Counter
from pathlib import Path
SCRATCH = Path(os.environ.get('SCRATCH') or '/SCRATCH-not-set')  # Claude Code scratch folder of the Oct 6 session that made the intermediate files (outside the repo); set SCRATCH to it

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/llm-sentence-edits-20261005'))
SCR = (SCRATCH / 'claude-fable')
HOSTED = ROOT / 'research/claude-hosted-edits-20261006'
SECTIONS = ROOT / 'research/claude-sections-20261006'
GEN = 'claude-fable-5-1'
N = {'polish': 300, 'humanizer': 200, 'cowrite': 100}
BATCH = {'polish': 50, 'humanizer': 50, 'cowrite': 25}
NEVER_TRAIN = HERE / 'heldout-never-train.json'
TOK = re.compile(r'\S+\s*')


def nh(t):
    return hashlib.sha256(re.sub(r'\W+', '', t.lower()).encode()).hexdigest()


def register(paras):
    old = json.loads(NEVER_TRAIN.read_text()) if NEVER_TRAIN.exists() else {'paper_ids': [], 'paragraph_sha256': [], 'paragraph_shingle_sha256_12': []}
    hs = {nh(p) for p in paras if len(re.sub(r'\W+', '', p)) >= 100}
    sh = {hashlib.sha256(n[i:i + 60].encode()).hexdigest()[:12] for p in paras for n in [re.sub(r'\W+', '', p.lower())] if len(n) >= 100 for i in range(0, len(n) - 59, 10)}
    NEVER_TRAIN.write_text(json.dumps({'purpose': 'Fable 5.1 evaluation-only sets (polish, humanizer, cowrite): generated text. Never train on these.',
                                       'updated_pdt': time.strftime('%Y-%m-%d %H:%M', time.localtime()),
                                       'paper_ids': old['paper_ids'], 'paragraph_sha256': sorted(set(old['paragraph_sha256']) | hs),
                                       'paragraph_shingle_sha256_12': sorted(set(old['paragraph_shingle_sha256_12']) | sh)}, separators=(',', ':')))


def write_batches(job, recs):
    d = SCR / job; d.mkdir(parents=True, exist_ok=True); (d / 'outputs').mkdir(exist_ok=True)
    for b in range(0, len(recs), BATCH[job]):
        with open(d / f'fable-{job}-{b // BATCH[job] + 1:03d}.jsonl', 'w') as f:
            for r in recs[b:b + BATCH[job]]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


def prepare():
    from build_llm_edits import sentences
    rng = random.Random(20261013)
    hosts = [h for h in map(json.loads, open(HOSTED / 'items-e.unrun.jsonl')) if h['split'] == 'heldout']
    rng.shuffle(hosts)
    pol = [h for h in hosts if len(sentences(h['target'])) >= 4][:N['polish']]
    used = {h['host_id'] for h in pol}
    cow = [h for h in hosts if h['host_id'] not in used and h['prev'] and h['next'] and len(h['prev']) >= 300 and len(h['next']) >= 300][:N['cowrite']]
    items = {'polish': [], 'cowrite': [], 'humanizer': []}
    for h in pol:
        items['polish'].append({'id': f"polish/{h['host_id']}", 'host_id': h['host_id'], 'record_id': h['record_id'], 'source': h['source'],
                                'prev': h['prev'], 'paragraph': h['target'], 'next': h['next']})
    for h in cow:
        items['cowrite'].append({'id': f"cowrite/{h['host_id']}", 'host_id': h['host_id'], 'record_id': h['record_id'], 'source': h['source'],
                                 'document': '\n\n'.join([h['prev'], h['target'], h['next']])})
    secs = [r for r in map(json.loads, gzip.open(SECTIONS / 'claude-sections-v1-heldout.jsonl.gz', 'rt'))]
    ai = {r['item_id']: r for r in secs if r['generator']}; hum = {r['item_id']: r for r in secs if not r['generator']}
    cand = []
    for iid, r in ai.items():
        g = [q for q in r['regions'] if q['label'] == 1][0]; pos = g['start']
        for k, p in enumerate(r['text'][g['start']:g['end']].split('\n\n')):
            if len(p) >= 300 and iid in hum:
                cand.append((iid, k, pos, p))
            pos += len(p) + 2
    rng.shuffle(cand); per = Counter()
    for iid, k, pos, p in cand:
        if len(items['humanizer']) >= N['humanizer'] or per[iid] >= 2:
            continue
        per[iid] += 1; r = ai[iid]
        items['humanizer'].append({'id': f'humanizer/{iid}/p{k}', 'item_id': iid, 'paper_id': r['paper_id'], 'section_role': r['section_role'],
                                   'ai_generator': r['generator'], 'span_start': pos, 'span_end': pos + len(p), 'ai_paragraph': p,
                                   'prev': r['text'][max(0, pos - 1500):pos].split('\n\n', 1)[-1] if pos else '',
                                   'next': r['text'][pos + len(p):pos + len(p) + 1500].rsplit('\n\n', 1)[0]})
    for job, xs in items.items():
        with open(HERE / f'items-{job}.jsonl', 'w') as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
    write_batches('polish', [{'id': x['id'], 'prev_context': x['prev'], 'paragraph': x['paragraph'], 'next_context': x['next'],
                              'paragraph_sentences': len(sentences(x['paragraph']))} for x in items['polish']])
    write_batches('cowrite', [{'id': x['id'], 'document': x['document']} for x in items['cowrite']])
    write_batches('humanizer', [{'id': x['id'], 'prev_context': x['prev'], 'paragraph': x['ai_paragraph'], 'next_context': x['next']} for x in items['humanizer']])
    print(json.dumps({k: len(v) for k, v in items.items()} | {'humanizer_sections': len({x['item_id'] for x in items['humanizer']}),
                      'humanizer_source_writers': dict(Counter(x['ai_generator'] for x in items['humanizer']))}))


# ---------- labeling ----------

def word_diff_regions(orig, new):
    """Character labels for `new` by word-level diff against `orig`: tokens (word + trailing whitespace) that appear in an
    'equal' block are human (0); tokens in 'replace'/'insert' blocks are AI (1). Deleted original text has no characters
    in `new`, so it gets no label. Adjacent same-label characters are merged into regions."""
    a = [m.group() for m in TOK.finditer(orig)]; b = [m.group() for m in TOK.finditer(new)]
    key = lambda t: t.rstrip()
    labels = []
    for tag, i0, i1, j0, j1 in difflib.SequenceMatcher(a=[key(t) for t in a], b=[key(t) for t in b], autojunk=False).get_opcodes():
        for t in b[j0:j1]:
            labels.append((len(t), 0 if tag == 'equal' else 1))
    lead = len(new) - len(new.lstrip())
    regions, pos = [], lead
    for n, lab in labels:
        if regions and regions[-1]['label'] == lab:
            regions[-1]['end'] = pos + n
        else:
            regions.append({'start': pos, 'end': pos + n, 'label': lab})
        pos += n
    if lead:
        regions.insert(0, {'start': 0, 'end': lead, 'label': regions[0]['label'] if regions else 0})
    return regions, sum(1 for _, l in labels if l) / max(1, len(labels))


def read_outputs(job, key):
    got, bad = {}, 0
    for f in sorted((SCR / job / 'outputs').glob('*.jsonl')):
        for l in open(f):
            if not l.strip():
                continue
            try:
                x = json.loads(l)
            except ValueError:
                bad += 1; continue
            if x.get('id') and x['id'] not in got:
                got[x['id']] = x.get(key)
    return got, bad


def ingest(job):
    from build_llm_edits import sentences, sim
    items = {x['id']: x for x in map(json.loads, open(HERE / f'items-{job}.jsonl'))}
    rows, rej, gen_paras = [], Counter(), []
    if job == 'polish':
        got, bad = read_outputs(job, 'paragraph'); rej['bad_json_line'] += bad
        for iid, para in got.items():
            x = items.get(iid)
            if not x or not para or re.search(r'\n\s*\n', para.strip()):
                rej['missing_or_multi_paragraph'] += 1; continue
            para = re.sub(r'\s+', ' ', para).strip(); orig = x['paragraph']
            regs, frac = word_diff_regions(orig, para)
            if para == orig or frac == 0:
                rej['unchanged'] += 1; continue
            if frac > .45:
                rej['too_heavy (>45% tokens changed)'] += 1; continue
            if not .85 <= len(para) / len(orig) <= 1.15:
                rej['length_change >15%'] += 1; continue
            S = sentences(orig); touched = 0
            for a, b in S:  # a sentence counts as touched if any word token of it is missing from the new paragraph's equal blocks
                seg = orig[a:b]
                if seg.strip() not in para:
                    touched += 1
            if touched < .5 * len(S):
                rej['fewer than half the sentences touched'] += 1; continue
            if abs(len(sentences(para)) - len(S)) > 1:
                rej['sentence count changed by >1'] += 1; continue
            pre = x['prev'] + '\n\n' if x['prev'] else ''; post = '\n\n' + x['next'] if x['next'] else ''
            text = pre + para + post; off = len(pre)
            regions = ([{'start': 0, 'end': off, 'label': 0}] if off else []) + [{'start': r['start'] + off, 'end': r['end'] + off, 'label': r['label']} for r in regs] \
                + ([{'start': off + len(para), 'end': len(text), 'label': 0}] if post else [])
            base = {'record_id': x['record_id'], 'host_id': x['host_id'], 'source': x['source'], 'split': 'heldout', 'eval_set': 'light-polish-v1'}
            rows.append({'id': iid + '/fable', **base, 'slot': 'mixed', 'label': 'mixed', 'condition': 'light_polish', 'writer': GEN, 'text': text,
                         'regions': regions, 'target_start': off, 'target_end': off + len(para), 'ai_token_fraction': round(frac, 3)})
            orig_text = pre + orig + post
            rows.append({'id': iid + '/original', **base, 'slot': 'human_controls', 'label': 'human', 'condition': 'original', 'writer': 'human',
                         'text': orig_text, 'regions': [{'start': 0, 'end': len(orig_text), 'label': 0}]})
            gen_paras.append(para)
    elif job == 'humanizer':
        got, bad = read_outputs(job, 'paragraph'); rej['bad_json_line'] += bad
        secs = {}
        for r in map(json.loads, gzip.open(SECTIONS / 'claude-sections-v1-heldout.jsonl.gz', 'rt')):
            secs[(r['item_id'], bool(r['generator']))] = r
        for iid, para in got.items():
            x = items.get(iid)
            if not x or not para or re.search(r'\n\s*\n', para.strip()):
                rej['missing_or_multi_paragraph'] += 1; continue
            para = re.sub(r'\s+', ' ', para).strip()
            if not .6 <= len(para) / len(x['ai_paragraph']) <= 1.5:
                rej['length_out_of_range'] += 1; continue
            s_ = sim(x['ai_paragraph'], para)
            if s_ > .9:
                rej['too_close_to_ai_original'] += 1; continue
            if s_ < .1:
                rej['meaning_lost'] += 1; continue
            src = secs[(x['item_id'], True)]; t = src['text']
            text = t[:x['span_start']] + para + t[x['span_end']:]; d = len(para) - (x['span_end'] - x['span_start'])
            ai = [q for q in src['regions'] if q['label'] == 1][0]
            regions = [q for q in ({'start': 0, 'end': ai['start'], 'label': 0}, {'start': ai['start'], 'end': ai['end'] + d, 'label': 1},
                                   {'start': ai['end'] + d, 'end': len(text), 'label': 0}) if q['end'] > q['start']]
            base = {'paper_id': x['paper_id'], 'item_id': x['item_id'], 'section_role': x['section_role'], 'split': 'heldout', 'eval_set': 'humanizer-v1'}
            rows.append({'id': iid + '/humanized', **base, 'slot': 'mixed', 'label': 'mixed', 'condition': 'humanized', 'writer': GEN,
                         'source_ai_writer': x['ai_generator'], 'text': text, 'regions': regions,
                         'humanized_start': x['span_start'], 'humanized_end': x['span_start'] + len(para), 'similarity_to_ai_original': round(s_, 3)})
            rows.append({'id': iid + '/ai_original', **base, 'slot': 'mixed', 'label': 'mixed', 'condition': 'ai_before_humanizing', 'writer': x['ai_generator'],
                         'text': t, 'regions': src['regions']})
            h = secs[(x['item_id'], False)]
            rows.append({'id': iid + '/human_control', **base, 'slot': 'human_controls', 'label': 'human', 'condition': 'original', 'writer': 'human',
                         'text': h['text'], 'regions': h['regions']})
            gen_paras.append(para)
    else:  # cowrite
        got, bad = read_outputs(job, 'segments'); rej['bad_json_line'] += bad
        for iid, segs in got.items():
            x = items.get(iid)
            if not x or not isinstance(segs, list) or not segs:
                rej['missing'] += 1; continue
            doc = x['document']; norm_doc = re.sub(r'\s+', ' ', doc)
            text, regs, prov, ok = '', [], [], True
            for sg in segs:
                o, t = sg.get('origin'), sg.get('text', '')
                if o not in ('human', 'ai', 'ai_edited') or not t:
                    ok = False; rej['bad_segment'] += 1; break
                if o == 'human' and re.sub(r'\s+', ' ', t).strip() not in norm_doc:
                    ok = False; rej['human segment not verbatim in original'] += 1; break
                if o != 'human' and len(t.strip()) > 40 and re.sub(r'\s+', ' ', t).strip() in norm_doc:
                    ok = False; rej['ai segment copies original'] += 1; break
                s0 = len(text); text += t
                prov.append({'start': s0, 'end': len(text), 'origin': o})
            if not ok:
                continue
            ai_chars = sum(p['end'] - p['start'] for p in prov if p['origin'] != 'human')
            if not .1 <= ai_chars / max(1, len(text)) <= .9:
                rej['ai share outside 10-90%'] += 1; continue
            for p in prov:
                lab = 0 if p['origin'] == 'human' else 1
                if regs and regs[-1]['label'] == lab and regs[-1]['end'] == p['start']:
                    regs[-1]['end'] = p['end']
                else:
                    regs.append({'start': p['start'], 'end': p['end'], 'label': lab})
            base = {'record_id': x['record_id'], 'host_id': x['host_id'], 'source': x['source'], 'split': 'heldout', 'eval_set': 'cowrite-v1'}
            rows.append({'id': iid + '/fable', **base, 'slot': 'mixed', 'label': 'mixed', 'condition': 'cowrite_3_rounds', 'writer': GEN, 'text': text,
                         'regions': regs, 'provenance': prov, 'ai_char_fraction': round(ai_chars / len(text), 3)})
            rows.append({'id': iid + '/original', **base, 'slot': 'human_controls', 'label': 'human', 'condition': 'original', 'writer': 'human',
                         'text': doc, 'regions': [{'start': 0, 'end': len(doc), 'label': 0}]})
            gen_paras += [p_ for p_ in text.split('\n\n')]
    with gzip.open(HERE / f'{job}-eval-v1.jsonl.gz', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if gen_paras:
        register(gen_paras)
    print(json.dumps({'job': job, 'assigned': len(items), 'returned': len(got), 'accepted': sum(1 for r in rows if r['slot'] == 'mixed' and r['writer'] == GEN),
                      'rejects': dict(rej)}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare'); p = sub.add_parser('ingest'); p.add_argument('job', choices=list(N))
    a = ap.parse_args()
    prepare() if a.cmd == 'prepare' else ingest(a.job)
