"""Space-side API writer for hosted small edits (job d) and whole-paragraph edits (job e) through OpenRouter.

Same writer instructions as the Claude/Codex runs (D-/E-INSTRUCTIONS.md, minus their file/ingest sections) and the same
validation (build_hosted.check_d / check_e), one item per request, one retry that quotes the rejection reason.
Resumable: outputs are appended per item to --out/outputs.jsonl and done ids are skipped on rerun.

  python api_writer.py --job e --writer luna     --items items.jsonl --out DIR [--concurrency 48] [--budget-usd 20]
  python api_writer.py --job d --writer deepseek --items items.jsonl --out DIR

Writers:
  luna      openai/gpt-6-luna on OpenRouter flex (provider openai/flex, service_tier flex), as research/synthetic-mirrors/generator.json
  deepseek  deepseek/deepseek-v4.1-flash, provider order Baseten then DeepInfra only, Baseten retried first on every request
  deepseek-hf, glm-hf   the same models through Hugging Face Inference Providers (billed to Alice's personal HF credits)
Env: OPENROUTER_API_KEY (OpenRouter writers) or HF_TOKEN (the -hf writers; needs the Inference Providers permission). Writes accepted.jsonl (training rows; job e also writes the matched human rows) and stats.json.
"""
import argparse, asyncio, json, os, re, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent; REPO = HERE / 'repo'
for d in ('research/llm-sentence-edits-20261005', 'research/claude-hosted-edits-20261006'): sys.path.insert(0, str(REPO / d))
import build_hosted as bh  # noqa: E402

WRITERS = {
    'luna': {'model': 'openai/gpt-6-luna', 'generator': 'openai/gpt-6-luna', 'service_tier': 'flex', 'reasoning': {'effort': 'low', 'exclude': True},
             'provider': {'only': ['openai/flex'], 'allow_fallbacks': False, 'require_parameters': True}},
    'deepseek': {'model': 'deepseek/deepseek-v4.1-flash', 'generator': 'deepseek/deepseek-v4.1-flash',
                 'provider': {'order': ['baseten', 'deepinfra'], 'only': ['baseten', 'deepinfra'], 'allow_fallbacks': False}},
    # Evaluation-only writers (never in training): unseen open-weight families.
    'glm': {'model': 'z-ai/glm-5.3-flash', 'generator': 'z-ai/glm-5.3-flash', 'reasoning': {'effort': 'low', 'exclude': True},
            'provider': {'order': ['relace'], 'allow_fallbacks': True}},
    'mimo': {'model': 'xiaomi/mimo-v2.6-flash', 'generator': 'xiaomi/mimo-v2.6-flash', 'provider': {'order': ['darkbloom'], 'allow_fallbacks': True}},
    'spacebunny': {'model': 'stealth/space-bunny-alpha', 'generator': 'stealth/space-bunny-alpha', 'provider': {'allow_fallbacks': True}},
    # Hugging Face Inference Providers (Oct 7, Alice): billed to her personal account credits. OpenAI-compatible router; the
    # provider is a model-id suffix, so the fallback is a second model id tried when the first errors.
    'deepseek-hf': {'api': 'hf', 'models': ['deepseek-ai/DeepSeek-V4.1-Flash:deepinfra', 'deepseek-ai/DeepSeek-V4.1-Flash:novita'],
                    'generator': 'deepseek/deepseek-v4.1-flash', 'price': {'deepinfra': (0.20, 0.60), 'novita': (0.30, 1.20)}},
    'glm-hf': {'api': 'hf', 'models': ['zai-org/GLM-5.3-Flash:together', 'zai-org/GLM-5.3-Flash:baseten'],
               'generator': 'z-ai/glm-5.3-flash', 'price': {'together': (0.15, 0.50), 'baseten': (0.15, 0.50)}},
}
API = {'openrouter': ('https://openrouter.ai/api/v1/chat/completions', 'OPENROUTER_API_KEY'),
       'hf': ('https://router.huggingface.co/v1/chat/completions', 'HF_TOKEN')}
for w, c in WRITERS.items():
    bh.MODEL[w] = c['generator']; bh.WRITTEN_BY[w] = 'hf-inference-providers' if c.get('api') == 'hf' else 'openrouter-api'
KEY = {'d': 'edited_paragraph', 'e': 'paragraph'}


def instructions(job):
    text = (REPO / f'research/openai-writer-handoff-20261006/{job.upper()}-INSTRUCTIONS.md').read_text()
    text = re.sub(r'^> \*\*Codex handoff note.*?\n\n', '', text, flags=re.S)
    text = re.split(r'\n## Ingest', text)[0]
    text = re.sub(r'\n- Write one output file per batch file.*', '', text)
    key = KEY[job]
    return (text + f'\n\n## This run\n\nYou receive exactly one input item (one JSON object) per message. Reply with exactly one JSON object '
            f'{{"id": "<the input id>", "{key}": "..."}} and nothing else: no commentary, no Markdown, no code fences.')


def parse(content, key):
    s = (content or '').strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    try:
        return json.loads(s).get(key)
    except Exception:
        m = re.search(r'\{.*\}', s, re.S)
        if m:
            try: return json.loads(m.group(0)).get(key)
            except Exception: return None
    return None


def hf_cost(cfg, resp, model):
    """The HF router reports token usage but no price: estimate it from the provider's listed price per 1M tokens."""
    u = resp.get('usage') or {}; pin, pout = cfg['price'][model.rsplit(':', 1)[1]]
    return (u.get('prompt_tokens', 0) * pin + u.get('completion_tokens', 0) * pout) / 1e6


async def call(session_post, cfg, messages, max_tokens):
    if cfg.get('api') == 'hf':  # each request tries the first provider, then the fallback
        for attempt in range(6):
            for model in cfg['models']:
                status, resp = await session_post({'model': model, 'messages': messages, 'max_tokens': max_tokens})
                if status == 200 and resp.get('choices'):
                    resp['provider'] = model.rsplit(':', 1)[1]; resp.setdefault('usage', {})['cost'] = hf_cost(cfg, resp, model)
                    return resp
            await asyncio.sleep(min(60, 5 * 2 ** attempt))
        return {'error': resp if isinstance(resp, dict) else str(resp), 'status': status}
    body = {'model': cfg['model'], 'messages': messages, 'max_tokens': max_tokens, 'provider': cfg['provider'], 'usage': {'include': True}}
    for k in ('service_tier', 'reasoning'):
        if k in cfg: body[k] = cfg[k]
    for attempt in range(6):
        status, resp = await session_post(body)
        if status == 200 and resp.get('choices'):
            return resp
        await asyncio.sleep(min(60, 5 * 2 ** attempt))
    return {'error': resp if isinstance(resp, dict) else str(resp), 'status': status}


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', choices=['d', 'e'], required=True); ap.add_argument('--writer', choices=list(WRITERS), required=True)
    ap.add_argument('--items', required=True); ap.add_argument('--out', required=True); ap.add_argument('--concurrency', type=int, default=48)
    ap.add_argument('--budget-usd', type=float, default=20.0); ap.add_argument('--limit', type=int)
    a = ap.parse_args(); cfg = WRITERS[a.writer]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import requests
    url, key_env = API[cfg.get('api', 'openrouter')]
    def post(body):
        try:
            r = requests.post(url, json=body, timeout=(20, 600),
                              headers={'Authorization': 'Bearer ' + os.environ[key_env], 'Content-Type': 'application/json'})
            return r.status_code, r.json()
        except Exception as e:
            return 0, {'error': repr(e)[:200]}
    async def session_post(body): return await asyncio.to_thread(post, body)
    items = [json.loads(l) for l in open(a.items)]
    if a.limit: items = items[:a.limit]
    done = {json.loads(l)['host_id'] for l in open(out / 'outputs.jsonl')} if (out / 'outputs.jsonl').exists() else set()
    todo = [h for h in items if h['host_id'] not in done]
    system = instructions(a.job); key = KEY[a.job]; spent = [0.0]; stats = Counter(); providers = Counter(); lock = asyncio.Lock()
    fo = open(out / 'outputs.jsonl', 'a'); fa = open(out / 'accepted.jsonl', 'a'); sem = asyncio.Semaphore(a.concurrency); t0 = time.time()

    async def one(h):
        async with sem:
            if spent[0] >= a.budget_usd: stats['skipped_budget'] += 1; return
            h = {**h, 'writer': a.writer}
            rec = bh.batch_rec(a.job, h); msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(rec, ensure_ascii=False)}]
            reason = None; result = None; cost = 0.0; prov = None
            for attempt in range(2):
                resp = await call(session_post, cfg, msgs, 4096)
                if 'error' in resp: reason = 'api_error'; break
                cost += float((resp.get('usage') or {}).get('cost') or 0); prov = resp.get('provider')
                content = resp['choices'][0]['message'].get('content'); text = parse(content, key)
                if text is None: reason = 'unparseable'
                else:
                    result, reason = (bh.check_d(h, text) if a.job == 'd' else bh.check_e(h, text))
                    if result: break
                msgs += [{'role': 'assistant', 'content': content or ''},
                         {'role': 'user', 'content': f'That output was rejected by the automatic check: {reason}. Redo the same item following the instructions exactly. Reply with the JSON object only.'}]
            async with lock:
                spent[0] += cost; providers[prov] += 1; stats['accepted' if result else 'rejected:' + str(reason)] += 1
                fo.write(json.dumps({'host_id': h['host_id'], 'ok': bool(result), 'reason': reason, 'cost': cost, 'provider': prov}) + '\n'); fo.flush()
                if result:
                    for r in (result if isinstance(result, list) else [result]):
                        fa.write(json.dumps({**r, 'writer': a.writer, 'split': h.get('split', 'train')}, ensure_ascii=False) + '\n')
                    fa.flush()
                n = sum(v for k, v in stats.items() if k != 'skipped_budget')
                if n % 50 == 0:
                    (out / 'stats.json').write_text(json.dumps({'done': n, 'of': len(todo), 'spent_usd': round(spent[0], 4), 'seconds': round(time.time() - t0),
                                                                'stats': dict(stats), 'providers': dict(providers)}, indent=1))
    await asyncio.gather(*(one(h) for h in todo))
    res = {'done': sum(v for k, v in stats.items() if k != 'skipped_budget'), 'of': len(todo), 'spent_usd': round(spent[0], 4),
           'seconds': round(time.time() - t0), 'stats': dict(stats), 'providers': dict(providers), 'finished': True}
    (out / 'stats.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    asyncio.run(main())
