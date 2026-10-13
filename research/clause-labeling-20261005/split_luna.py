"""Split paired paper passages into clauses with GPT-6 Luna (OpenRouter, Flex tier).

First step of the soft n-gram labeling test (Pangram 4 report, section 3.5): the report
uses Claude Haiku 4.5 as its clause splitter; we use Luna with the repo's existing Flex
configuration (research/synthetic-mirrors/generator.json).

The model returns clauses as verbatim substrings; offsets are recovered by in-order search
and every split must tile its passage exactly (gaps may contain only whitespace).
Rejected splits are recorded, never repaired. The key is read from
~/.config/open-text-detector/openrouter_key into memory and never printed or stored.

Usage: split_luna.py DATASET OUT_DIR --pairs 25 [--budget-usd 1.0] [--workers 8]
Each pair contributes two passages: the human source paragraph and the Luna rewrite.
With --items, DATASET is instead a JSONL of {gold_id, text} rows (e.g. the gold sentences).
"""
import argparse, ast, hashlib, json, os, random, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

GEN = json.loads((Path(__file__).resolve().parents[1] / 'synthetic-mirrors' / 'generator.json').read_text())
URL = 'https://openrouter.ai/api/v1/chat/completions'
PROMPT_VERSION = 'clause-split-v1'
SYSTEM = (
    'You split text into clauses. A clause is a group of words containing a subject and a verb; '
    'treat each independent clause and each subordinate or relative clause that expresses its own idea '
    'as a separate clause. Fragments without a verb (headings, list items, citations, equations, '
    'parenthetical asides) attach to the neighbouring clause they belong to. '
    'Copy the text exactly: every character, including punctuation, citations, math and odd spacing, '
    'must appear in exactly one clause, in the original order. Do not fix, normalise or paraphrase anything. '
    'Return JSON: {"clauses": ["...", "..."]}.'
)


def key():
    return Path('~/.config/open-text-detector/openrouter_key').expanduser().read_text().strip()


def body(text):
    b = {k: GEN[k] for k in ('model', 'provider', 'service_tier', 'reasoning')}
    b.update(messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
             response_format={'type': 'json_object'}, max_tokens=min(8192, 1024 + 2 * len(text) // 3))
    return b


def tile(text, clauses):
    """Return [(start, end)] if clauses tile text in order with whitespace-only gaps, else an error string."""
    spans, pos = [], 0
    for c in clauses:
        if not isinstance(c, str) or not c.strip():
            return 'empty_or_nonstring_clause'
        c = c.strip()
        i = text.find(c, pos)
        if i < 0:
            return 'clause_not_verbatim_in_order'
        if text[pos:i].strip():
            return 'non_whitespace_gap'
        spans.append((i, i + len(c))); pos = i + len(c)
    if text[pos:].strip():
        return 'uncovered_tail'
    return spans


def call(text, auth, log):
    req = json.dumps(body(text)).encode()
    for attempt in range(4):
        r = urllib.request.Request(URL, data=req, headers={'Authorization': 'Bearer ' + auth, 'Content-Type': 'application/json'})
        t0 = time.time()
        try:
            with urllib.request.urlopen(r, timeout=900) as x:
                return json.load(x), attempt, time.time() - t0
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                detail = json.load(e).get('error', {}).get('message', '')[:200]
            except Exception:
                detail = ''
        except (urllib.error.URLError, TimeoutError) as e:
            status, detail = 0, type(e).__name__
        log(f'  http {status} attempt {attempt + 1}: {detail}')
        if status in (400, 401, 402, 403, 404):
            break
        time.sleep(15 * 2 ** attempt)
    return {'error': {'status': status, 'detail': detail}}, attempt, None


def load_pairs(path, n, seed):
    by = {}
    for line in open(path):
        r = json.loads(line)
        by.setdefault(r['passage_id'], {})[r['operation']] = r
    ids = sorted(p for p, v in by.items() if {'human_original', 'paragraph_generate'} <= v.keys())
    random.Random(seed).shuffle(ids)
    out = []
    for pid in ids[:n]:
        h, g = by[pid]['human_original'], by[pid]['paragraph_generate']
        reg = next(x for x in (g["regions"] if isinstance(g["regions"], list) else ast.literal_eval(g["regions"])) if x['label'] == 'ai_rewritten')
        out.append({'passage_id': pid, 'paper_id': g['paper_id'],
                    'source': h['text'][reg['source_start']:reg['source_end']],
                    'target': g['text'][reg['start']:reg['end']]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--pairs', type=int); ap.add_argument('--items', action='store_true'); ap.add_argument('--seed', type=int, default=20261005)
    ap.add_argument('--budget-usd', type=float, default=1.0); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    done_path = a.out / 'splits.jsonl'
    done = {json.loads(l)['item_id'] for l in open(done_path)} if done_path.exists() else set()
    if a.items:
        items = [dict(item_id=r['gold_id'], text=r['text']) for r in map(json.loads, open(a.dataset))]
    else:
        items = [dict(item_id=f"{p['passage_id']}/{side}", passage_id=p['passage_id'], paper_id=p['paper_id'], side=side, text=p[side])
                 for p in load_pairs(a.dataset, a.pairs, a.seed) for side in ('source', 'target')]
    items = [i for i in items if i['item_id'] not in done]
    auth, lock, spent, stop = key(), threading.Lock(), [0.0], threading.Event()
    progress = open(a.out / 'progress.log', 'a')

    def log(msg):
        with lock:
            progress.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n"); progress.flush()

    log(f'start: {len(items)} passages pending, {len(done)} already done, budget ${a.budget_usd}')

    def work(it):
        if stop.is_set():
            return
        resp, attempts, secs = call(it['text'], auth, log)
        rec = dict(it, prompt_version=PROMPT_VERSION, requested_model=GEN['model'], requested_service_tier=GEN['service_tier'],
                   attempts=attempts + 1, seconds=secs, saved_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   text_sha256=hashlib.sha256(it['text'].encode()).hexdigest())
        if 'error' in resp:
            rec.update(state='api_error', error=resp['error'])
        else:
            tier = resp.get('service_tier')
            rec.update(model=resp.get('model'), reported_service_tier=tier,
                       tier_verification='verified' if tier == 'flex' else 'unreported' if tier is None else 'unexpected',
                       usage=resp.get('usage'), finish_reason=resp['choices'][0].get('finish_reason'),
                       raw=resp['choices'][0]['message'].get('content'))
            try:
                clauses = json.loads(rec['raw'])['clauses']
                spans = tile(it['text'], clauses)
            except (ValueError, KeyError, TypeError) as e:
                spans = 'unparseable_json'
            if isinstance(spans, str):
                rec.update(state='rejected', reject_reason=spans)
            else:
                rec.update(state='ok', spans=spans)
        cost = float((rec.get('usage') or {}).get('cost') or 0)
        with lock:
            spent[0] += cost
            with open(done_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if spent[0] >= a.budget_usd:
                stop.set()
        log(f"{rec['state']} {rec['item_id']} {rec.get('reject_reason', '')} tier={rec.get('reported_service_tier')} cost=${cost:.5f} total=${spent[0]:.4f}")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, items))
    log(f"end: spent ${spent[0]:.4f}{' (budget stop)' if stop.is_set() else ''}")


if __name__ == '__main__':
    main()
