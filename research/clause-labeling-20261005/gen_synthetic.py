"""Build source/edited paragraph pairs with known per-character provenance.

Each human paragraph S (a pre-2022 paper paragraph) gets an edit budget: none (control),
1 sentence, 2 sentences, or about half its sentences. Each edited sentence gets one op,
applied by GPT-6 Luna (OpenRouter Flex, research/synthetic-mirrors/generator.json):
  polish      light wording/grammar changes, same sentence   -> truth AI-Assisted
  paraphrase  same content, substantially different wording  -> truth AI-Assisted
  insert      a new sentence with a new point, placed after  -> truth AI-Generated
  clause_paraphrase  one clause of a long sentence reworded    -> that clause AI-Assisted
  clause_append      a new clause added to a human sentence    -> that clause AI-Generated
Clause ops mark their edit with \u27e6\u27e7; the edited span is then taken from the common
prefix/suffix of original and result (an append must be a pure insertion), so truth is
exact at character level. They apply only to sentences of
at least CLAUSE_MIN_CHARS characters.
Untouched text is Human. We assemble T ourselves from the returned sentences, so every
character of T has a known label. Edits are checked against their definition (word-level
similarity bands below); a paragraph with any failed edit is dropped, never repaired.

Usage: gen_synthetic.py DATASET OUT_DIR --n 300 [--budget-usd 1.0] [--workers 8] [--production]
       gen_synthetic.py LLM_EDITS.jsonl.gz OUT_DIR --candidates candidates-v1.jsonl --n 100000   (production pool B)
"""
import argparse, ast, difflib, hashlib, json, random, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

GEN = json.loads((Path(__file__).resolve().parents[1] / 'synthetic-mirrors' / 'generator.json').read_text())
URL = 'https://openrouter.ai/api/v1/chat/completions'
PROMPT_VERSION = 'synthetic-edits-v3'
OPS = {
    'polish': 'Lightly polish this sentence: improve grammar, word choice or flow with minimal changes. '
              'It must stay recognisably the same sentence; change only a few words.',
    'paraphrase': 'Rewrite this sentence so it says the same thing in substantially different words: change the sentence '
                  'structure and reuse as few of the original words as you can, apart from technical terms, names, '
                  'citations, numbers and symbols, which must all be kept.',
    'insert': 'Write ONE new sentence to place immediately after this sentence. It must add a new, plausible point '
              'that is not stated anywhere in the paragraph, in the same academic style.',
    'clause_paraphrase': 'Rewrite exactly ONE clause of this sentence (not the whole sentence) in substantially different '
                         'words, keeping its meaning, and leave every other character of the sentence unchanged. '
                         'Wrap the rewritten clause in \u27e6 and \u27e7.',
    'clause_append': 'Extend this sentence by adding ONE new clause that makes a new, plausible point not stated anywhere '
                     'in the paragraph (for example a ", which ..." or ", while ..." clause), leaving the original '
                     'wording unchanged. Wrap only the added words in \u27e6 and \u27e7.',
}
CLAUSE_OPS = ('clause_paraphrase', 'clause_append')
CLAUSE_MIN_CHARS = 80
SYSTEM = ('You edit sentences of an academic paragraph as instructed. Match the paragraph\'s register and conventions. '
          'Return JSON: {"edits": [{"i": <sentence number>, "op": "<op>", "text": "<the resulting sentence>"}]} '
          'with exactly one entry per instruction, in the same order. Return only the sentence text for each edit, '
          'including the \u27e6 \u27e7 markers where an instruction asks for them.')
# Word-level similarity (difflib ratio) bands that define each op.
BANDS = {'polish': (0.60, 0.985), 'paraphrase': (0.0, 0.60)}
INSERT_MAX_SIM = 0.50
BUDGETS = [('control', 0.15), ('one', 0.30), ('two', 0.30), ('half', 0.25)]
# Training data: no controls (the human originals already exist), weighted toward small edits.
PRODUCTION_BUDGETS = [('one', 0.45), ('two', 0.35), ('half', 0.20)]


def words(t):
    return re.findall(r'\w+|[^\w\s]', t.lower())


def sim(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def regions(g):
    return g['regions'] if isinstance(g['regions'], list) else ast.literal_eval(g['regions'])


def load_paragraphs(path, n, seed, production=False):
    by = {}
    for line in open(path):
        r = json.loads(line)
        by.setdefault(r['passage_id'], {})[r['operation']] = r
    paras = []
    for pid, v in sorted(by.items()):
        if not {'human_original', 'paragraph_generate'} <= v.keys():
            continue
        h = v['human_original']
        reg = next(x for x in regions(v['paragraph_generate']) if x['label'] == 'ai_rewritten')
        lo, hi = reg['source_start'], reg['source_end']
        sents = [(s['start'] - lo, s['end'] - lo) for s in h['sentences'] if s['start'] >= lo and s['end'] <= hi]
        text = h['text'][lo:hi]
        if 4 <= len(sents) <= 12 and all(20 <= b - a <= 500 for a, b in sents):
            paras.append({'passage_id': pid, 'paper_id': h['paper_id'], 'source': text, 'sentences': sents,
                          'source_split': h['split']})
    rng = random.Random(seed)
    rng.shuffle(paras)
    seen, out = set(), []
    if production:  # every eligible paragraph, minus never-train papers/paragraphs
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm-sentence-edits-20261005'))
        from build_llm_edits import never_train_hit
        out = [p for p in paras if not never_train_hit({'paper_id': p['paper_id'], 'text': p['source']})][:n]
    for p in ([] if production else paras):  # one paragraph per paper
        if p['paper_id'] not in seen:
            seen.add(p['paper_id']); out.append(p)
        if len(out) == n:
            break
    return assign_plans(out, seed, production)


def assign_plans(out, seed, production):
    for k, p in enumerate(out):
        r = random.Random(f'{seed}/{p["passage_id"]}')
        budgets = PRODUCTION_BUDGETS if production else BUDGETS
        budget = r.choices([b for b, _ in budgets], [w for _, w in budgets])[0]
        m = len(p['sentences'])
        k_edit = {'control': 0, 'one': 1, 'two': 2, 'half': max(2, m // 2)}[budget]
        idx = sorted(r.sample(range(m), k_edit))
        p['budget'] = budget
        p['plan'] = []
        for i in idx:
            long_enough = p['sentences'][i][1] - p['sentences'][i][0] >= CLAUSE_MIN_CHARS
            p['plan'].append({'i': i, 'op': r.choice(list(OPS) if long_enough else [o for o in OPS if o not in CLAUSE_OPS])})
        p['split'] = p['source_split'] if production else ('dev' if int(hashlib.sha256(p['paper_id'].encode()).hexdigest(), 16) % 2 == 0 else 'test')
    return out


SENT = re.compile(r'\S.*?(?:[.!?](?=\s|$)|$)', re.S)  # same sentence rule as build_llm_edits.py


def load_candidates(path, used_path, n, seed):
    """Unused screened paragraphs from the sentence-edit builder's pool (paired-train papers only).

    The archive_pre2023 rows are skipped: that pool waits for the extraction upgrade."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llm-sentence-edits-20261005'))
    from build_llm_edits import never_train_hit
    import gzip
    used = {json.loads(l)['id'].rsplit('/', 1)[0] for l in gzip.open(used_path, 'rt')}
    out = []
    for x in map(json.loads, open(path)):
        if x['cand_id'] in used or x['source'] != 'paired_train_nontarget':
            continue
        sents = [(m.start(), m.end()) for m in SENT.finditer(x['target']) if m.group().strip()]
        if not (4 <= len(sents) <= 12 and all(20 <= b - a <= 500 for a, b in sents)):
            continue
        if never_train_hit({'paper_id': x['paper_id'], 'text': '\n\n'.join([x['before'], x['target'], x['after']])}):
            continue
        out.append({'passage_id': x['cand_id'], 'paper_id': x['paper_id'], 'source': x['target'], 'sentences': sents,
                    'source_split': 'train', 'context_before': x['before'], 'context_after': x['after']})
    random.Random(seed).shuffle(out)
    return assign_plans(out[:n], seed, True)


def prompt(p):
    lines = [f'[{j + 1}] {p["source"][a:b]}' for j, (a, b) in enumerate(p['sentences'])]
    asks = [f'- sentence {e["i"] + 1}, op "{e["op"]}": {OPS[e["op"]]}' for e in p['plan']]
    return 'Paragraph, one numbered sentence per line:\n' + '\n'.join(lines) + '\n\nInstructions:\n' + '\n'.join(asks)


def post(messages, auth, max_tokens):
    b = {k: GEN[k] for k in ('model', 'provider', 'service_tier', 'reasoning')}
    b.update(messages=messages, response_format={'type': 'json_object'}, max_tokens=max_tokens)
    data = json.dumps(b).encode()
    last = None
    for attempt in range(5):
        req = urllib.request.Request(URL, data=data, headers={'Authorization': 'Bearer ' + auth, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=900) as x:
                body = json.load(x)
            if 'error' not in body:
                return body, attempt + 1
            status = int(body['error'].get('code') or 0)  # e.g. 502 "Flex processing is temporarily unavailable"
            last = body
        except urllib.error.HTTPError as e:
            status = e.code
        except (urllib.error.URLError, TimeoutError):
            status = 0
        if status in (400, 401, 402, 403, 404):
            break
        time.sleep(30 * 2 ** attempt)
    return {'error': dict(status=status, **(last.get('error', {}) if last else {}))}, attempt + 1


def check_clause(op, raw, orig, p):
    """Validate a clause edit; return (ok, similarity, [pre, mid, post]).

    The markers only ask the model to keep its edit local; the edited span itself comes
    from the common prefix and suffix of the original and the marker-free result, so the
    truth does not depend on where the model put the markers.
    """
    if raw.count('\u27e6') != 1 or raw.count('\u27e7') != 1:
        return False, -1.0, None
    new_s = ' '.join(raw.replace('\u27e6', '').replace('\u27e7', '').split())
    new_s = re.sub(r'\s+([,.;:!?])', r'\1', new_s)
    n = 0
    while n < min(len(orig), len(new_s)) and orig[n] == new_s[n]:
        n += 1
    m = 0
    while m < min(len(orig), len(new_s)) - n and orig[-1 - m] == new_s[-1 - m]:
        m += 1
    while 0 < n < len(new_s) and new_s[n - 1].isalnum() and new_s[n].isalnum():
        n -= 1  # never split a word at the start of the edit
    while m > 0 and new_s[len(new_s) - m - 1].isalnum() and new_s[len(new_s) - m].isalnum():
        m -= 1  # nor at its end
    pre, mid, post = new_s[:n], new_s[n:len(new_s) - m], new_s[len(new_s) - m:]
    old_mid = orig[n:len(orig) - m]
    if op == 'clause_append':
        ok = not old_mid and len(mid.strip()) >= 15 and not re.search(r'[.!?]\s*$', pre)
        s_ = max(sim(mid, p['source'][x:y]) for x, y in p['sentences'])
        return ok and s_ < INSERT_MAX_SIM, s_, [pre, mid, post]
    if not old_mid.strip() or not mid.strip():
        return False, -1.0, [pre, mid, post]
    s_ = sim(mid, old_mid)
    ok = len(old_mid.strip()) >= 15 and len(old_mid) / len(orig) <= 0.7 and s_ < 0.6
    return ok, s_, [pre, mid, post]


def assemble(p, edits):
    """Build T and its per-character truth spans from S, the sentence offsets and validated edits."""
    by_i = {e['i']: e for e in edits}
    s, sents = p['source'], p['sentences']
    parts, spans, pos = [], [], 0
    def emit(text, label, op, i):
        nonlocal pos
        parts.append(text); spans.append({'start': pos, 'end': pos + len(text), 'label': label, 'op': op, 'sentence': i}); pos += len(text)
    emit(s[:sents[0][0]], 'human', 'gap', None) if sents[0][0] else None
    for j, (a, b) in enumerate(sents):
        e = by_i.get(j)
        if e and e['op'] in CLAUSE_OPS:
            pre, mid, post = e['parts']
            emit(pre, 'human', 'keep', j)
            emit(mid, 'assisted' if e['op'] == 'clause_paraphrase' else 'generated', e['op'], j)
            emit(post, 'human', 'keep', j)
        elif e and e['op'] != 'insert':
            emit(e['text'], 'assisted', e['op'], j)
        else:
            emit(s[a:b], 'human', 'keep', j)
        if e and e['op'] == 'insert':
            emit(' ', 'human', 'gap', None); emit(e['text'], 'generated', 'insert', j)
        tail = s[b:sents[j + 1][0]] if j + 1 < len(sents) else s[b:]
        if tail:
            emit(tail, 'human', 'gap', None)
    return ''.join(parts), [x for x in spans if x['end'] > x['start']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--seed', type=int, default=20261006)
    ap.add_argument('--budget-usd', type=float, default=1.0); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--candidates', type=Path, help='use unused paragraphs from this candidates-v1.jsonl instead of DATASET')
    ap.add_argument('--production', action='store_true', help='all eligible paragraphs (not one per paper), source splits kept, '
                    'never-train rows dropped, no controls')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    out_path = a.out / 'pairs.jsonl'
    if out_path.exists():  # failed calls move to errors.jsonl so a rerun retries them
        rows = [json.loads(l) for l in open(out_path)]
        with open(a.out / 'errors.jsonl', 'a') as f:
            for r in rows:
                if r['state'] == 'api_error':
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(out_path, 'w') as f:
            for r in rows:
                if r['state'] != 'api_error':
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    done = {json.loads(l)['passage_id'] for l in open(out_path)} if out_path.exists() else set()
    paras = [p for p in (load_candidates(a.candidates, a.dataset, a.n, a.seed) if a.candidates else load_paragraphs(a.dataset, a.n, a.seed, a.production)) if p['passage_id'] not in done]
    auth = Path('~/.config/open-text-detector/openrouter_key').expanduser().read_text().strip()
    lock, spent, stop = threading.Lock(), [0.0], threading.Event()
    log = open(a.out / 'progress.log', 'a')

    def note(msg):
        with lock:
            log.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n"); log.flush()

    note(f'start: {len(paras)} paragraphs pending, {len(done)} done, budget ${a.budget_usd}')

    def work(p):
        if stop.is_set():
            return
        rec = dict(p, prompt_version=PROMPT_VERSION, requested_model=GEN['model'], requested_service_tier=GEN['service_tier'])
        cost = 0.0
        if not p['plan']:
            rec.update(state='ok', target=p['source'], truth=assemble(p, [])[1], edits=[])
        else:
            resp, attempts = post([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt(p)}], auth, 4096)
            rec['attempts'] = attempts
            if 'error' in resp:
                rec.update(state='api_error', error=resp['error'])
            else:
                tier = resp.get('service_tier'); cost = float((resp.get('usage') or {}).get('cost') or 0)
                rec.update(model=resp.get('model'), reported_service_tier=tier, usage=resp.get('usage'),
                           tier_verification='verified' if tier == 'flex' else 'unreported' if tier is None else 'unexpected',
                           raw=resp['choices'][0]['message'].get('content'))
                try:
                    got = json.loads(rec['raw'])['edits']
                    edits, reasons = [], []
                    if [(int(g['i']) - 1, g['op']) for g in got] != [(e['i'], e['op']) for e in p['plan']]:
                        reasons.append('edit_list_mismatch')
                    else:
                        for g, e in zip(got, p['plan']):
                            t = ' '.join(str(g['text']).split())
                            orig = p['source'][slice(*p['sentences'][e['i']])]
                            if e['op'] in CLAUSE_OPS:
                                ok, e_sim, parts = check_clause(e['op'], str(g['text']), orig, p)
                                edits.append(dict(e, text=''.join(parts) if parts else str(g['text']), parts=parts, similarity=round(e_sim, 4)))
                                if not ok:
                                    reasons.append(f"{e['op']}_invalid:{e_sim:.2f}")
                                continue
                            if e['op'] == 'insert':
                                ms = max(sim(t, p['source'][x:y]) for x, y in p['sentences'])
                                ok = ms < INSERT_MAX_SIM and len(t) >= 20
                                e_sim = ms
                            else:
                                e_sim = sim(t, orig); lo, hi = BANDS[e['op']]
                                ok = lo <= e_sim < hi and t != orig
                            edits.append(dict(e, text=t, similarity=round(e_sim, 4)))
                            if not ok:
                                reasons.append(f"{e['op']}_out_of_band:{e_sim:.2f}")
                    if reasons:
                        rec.update(state='rejected', reject_reasons=reasons, edits=edits)
                    else:
                        target, truth = assemble(p, edits)
                        rec.update(state='ok', target=target, truth=truth, edits=edits)
                except (ValueError, KeyError, TypeError):
                    rec.update(state='rejected', reject_reasons=['unparseable_json'])
        with lock:
            spent[0] += cost
            with open(out_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if spent[0] >= a.budget_usd:
                stop.set()
        note(f"{rec['state']} {p['passage_id']} {p['budget']} {','.join(rec.get('reject_reasons', []))} cost=${cost:.5f} total=${spent[0]:.4f}")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, paras))
    note(f"end: spent ${spent[0]:.4f}{' (budget stop)' if stop.is_set() else ''}")


if __name__ == '__main__':
    main()
