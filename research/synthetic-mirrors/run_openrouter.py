"""Luna mirror generation with bounded spending and crash-safe, no-replay calls."""
import argparse
import asyncio
import fcntl
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mirror_core import PROTOCOL, assess, parse_topic, sha, topic_messages, validate_parent, writer_messages

HERE = Path(__file__).resolve().parent
GENERATOR = json.loads((HERE / 'generator.json').read_text())
_DURABLE_STORE = None

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    if _DURABLE_STORE is not None:
        _DURABLE_STORE.persist(path)

def request(messages, stage, target_words=0):
    body = {key: GENERATOR[key] for key in ('model', 'provider', 'service_tier', 'reasoning')}
    body['messages'] = messages
    body['max_tokens'] = (GENERATOR['topic_max_tokens'] if stage == 'topic' else
                          min(GENERATOR['writer_max_tokens'], max(GENERATOR['writer_min_max_tokens'], target_words * 4)))
    if stage == 'topic':
        body['response_format'] = {'type': 'json_object'}
    return body

def reservation(body):
    # UTF-8 bytes conservatively bound text tokens; overhead covers chat framing.
    input_bound = len(canonical(body).encode('utf8')) + 2048
    if input_bound >= 272000:
        raise ValueError('Long-context price tier is outside this protocol')
    return (input_bound * GENERATOR['cache_write_ceiling_per_million'] +
            body['max_tokens'] * GENERATOR['provider']['max_price']['completion']) / 1e6

def decode(response):
    choice = response['choices'][0]
    text = choice['message'].get('content') or ''
    if not isinstance(text, str):
        raise ValueError('Unexpected non-text generation')
    usage = response['usage']
    reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    # Do not count hidden reasoning as generated passage text. Missing details
    # leave the length-token condition unverified, not automatically passed.
    visible = None if reasoning is None else usage['completion_tokens'] - reasoning
    return text.strip(), choice.get('finish_reason'), {
        'input_tokens': usage['prompt_tokens'], 'output_tokens': visible,
        'billing_completion_tokens': usage['completion_tokens'], 'reasoning_tokens': reasoning}

class BudgetExhausted(RuntimeError):
    pass

class BalanceGate:
    def __init__(self, rows, claimed, limits):
        self.limits = limits
        self.claimed = set(claimed)
        index = {r['record_id']: r for r in rows}
        if self.claimed - index.keys():
            raise RuntimeError('Existing calls refer to sources outside the frozen input')
        self.sources = Counter(index[i]['source_id'] for i in self.claimed)

    def claim(self, row):
        if row['record_id'] in self.claimed:
            return True
        if len(self.claimed) >= self.limits['max_documents']:
            return False
        quotas = self.limits.get('source_quotas')
        if quotas is not None and self.sources[row['source_id']] >= quotas.get(row['source_id'], 0):
            return False
        self.claimed.add(row['record_id'])
        self.sources[row['source_id']] += 1
        return True

class Client:
    def __init__(self, out, budget, fetch, canonical_model):
        self.out, self.budget, self.fetch, self.canonical_model = out, budget, fetch, canonical_model
        self.spent = 0.0
        self.uncertain_cost_reserve = 0.0
        self.inflight = 0.0
        self.condition = None
        self.halted = False
        for path in (out / 'calls').glob('*.json'):
            record = json.loads(path.read_text())
            if record.get('state') == 'interrupted_reserved_no_replay':
                self.uncertain_cost_reserve += record['reserved_usd']
                self.spent += record['reserved_usd']
                continue
            if record.get('state') != 'complete':
                raise RuntimeError('Unresolved dispatch; reconcile before resuming: ' + path.name)
            self.spent += record['cost_usd']
        if self.spent > budget:
            raise RuntimeError('Existing run already exceeds supplied budget')

    async def call(self, body, row_id, stage):
        call_id = sha(canonical({'request': body, 'parent': row_id, 'stage': stage}))
        path = self.out / 'calls' / (call_id + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['state'] != 'complete':
                raise RuntimeError('Unresolved call cannot be automatically replayed')
            return saved
        ceiling = reservation(body)
        if self.condition is None:
            self.condition = asyncio.Condition()
        async with self.condition:
            while self.spent + self.inflight + ceiling > self.budget:
                if self.halted:
                    raise RuntimeError('Another request needs reconciliation')
                if self.inflight <= 1e-12:
                    raise BudgetExhausted('Budget exhausted before next request; completed calls preserved')
                await self.condition.wait()
            if self.halted:
                raise RuntimeError('Another request needs reconciliation')
            self.inflight += ceiling
        record = {'call_id': call_id, 'source_record_id': row_id, 'stage': stage,
                  'request': body, 'reserved_usd': ceiling, 'state': 'dispatched',
                  'created_at': datetime.now(timezone.utc).isoformat()}
        try:
            await asyncio.to_thread(save, path, record)
            status, response = await self.fetch('chat/completions', body)
            record.update(http_status=status, response=response, state='needs_reconciliation')
            await asyncio.to_thread(save, path, record)
            if status != 200 or response.get('error'):
                raise RuntimeError('API failure preserved; no automatic paid retry')
            cost = response.get('usage', {}).get('cost')
            if cost is None or not math.isfinite(float(cost)) or float(cost) < 0:
                raise RuntimeError('Missing or invalid billed cost; reconcile before continuing')
            if response.get('model') not in (GENERATOR['model'], self.canonical_model):
                raise RuntimeError('Unexpected returned model')
            if response.get('service_tier') != GENERATOR['service_tier']:
                raise RuntimeError('Unexpected returned service tier')
            if float(cost) > ceiling:
                raise RuntimeError('Reported cost exceeded reservation; halted for reconciliation')
            record.update(state='complete', cost_usd=float(cost))
            await asyncio.to_thread(save, path, record)
            self.spent += float(cost)
            return record
        except BaseException:
            self.halted = True
            raise
        finally:
            async with self.condition:
                self.inflight -= ceiling
                self.condition.notify_all()


async def run(args, fetch=None):
    global _DURABLE_STORE
    from durable_store import configured_store
    _DURABLE_STORE = configured_store()
    if args.limit <= 0:
        raise ValueError('Limit must be positive')
    rows = [json.loads(line) for line in args.input.read_text().split('\n') if line.strip()][:args.limit]
    if not rows or len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError('Inputs must have unique, nonempty source identities')
    for row in rows:
        validate_parent(row, args.pilot)
    if not args.execute:
        print(json.dumps({'mode': 'dry_run', 'model': GENERATOR['model'], 'parents': len(rows),
                          'maximum_requests': len(rows) * 2, 'pilot': args.pilot,
                          'paid_requests': 0, 'budget_usd': args.budget_usd}))
        return
    if args.budget_usd is None or not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        raise ValueError('Execution requires an explicit positive --budget-usd')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await run_locked(args, rows, fetch)

async def run_locked(args, rows, fetch):
    if fetch is None:
        if os.environ.get('OPENROUTER_API_KEY'):
            import openrouter_transport as transport
        else:
            sys.path.insert(0, str(HERE.parents[1] / 'benchmarks/otd'))
            import arena100_generate as transport
            transport.REQUEST_TIMEOUT = 900
        fetch = transport.fetch
    status, endpoint = await fetch('models/' + GENERATOR['model'] + '/endpoints', auth=False)
    if status != 200:
        raise RuntimeError('Cannot verify current endpoint')
    flex = next(e for e in endpoint['data']['endpoints'] if e['tag'] == 'openai/flex')
    for key, ceiling in [('prompt', .05), ('completion', .25), ('input_cache_write', .0625)]:
        if float(flex['pricing'].get(key, 0)) * 1e6 > ceiling + 1e-10:
            raise RuntimeError('Endpoint price exceeds configured ceiling')
    status, catalog = await fetch('models', auth=False)
    if status != 200:
        raise RuntimeError('Cannot verify model revision')
    model = next(m for m in catalog['data'] if m['id'] == GENERATOR['model'])
    hosted_protocol = {**PROTOCOL, 'inference_dtype': GENERATOR['inference_dtype'],
                       'generator_diversity': 'User-selected Luna main generator; secondary generators not selected. No claim of reproducing Pangram generator weights.'}
    identity = {'input_sha256': sha(canonical(rows)), 'generator': GENERATOR, 'pilot': args.pilot,
                'protocol': hosted_protocol, 'canonical_model': model['canonical_slug'],
                'runner_sha256': sha(Path(__file__).read_text()),
                'core_sha256': sha((HERE / 'mirror_core.py').read_text())}
    manifest = args.out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError('Resume identity changed; use a separate run directory')
    save(manifest, identity)
    save(args.out / 'endpoint-latest.json', flex)
    client = Client(args.out, args.budget_usd, fetch, model['canonical_slug'])
    limits_file = args.out / 'limits.json'
    limits = json.loads(limits_file.read_text()) if limits_file.exists() else {'max_documents': len(rows)}
    if limits.get('budget_usd', args.budget_usd) < args.budget_usd:
        raise RuntimeError('Requested budget exceeds saved run limit')
    prior_calls = [json.loads(p.read_text()) for p in (args.out / 'calls').glob('*.json')]
    gate = BalanceGate(rows, [c['source_record_id'] for c in prior_calls], limits)
    records = []
    errors = []
    budget_reached = False
    stop = False
    async def process(row):
        dest = args.out / 'records' / (sha(row['record_id']) + '.json')
        if dest.exists():
            records.append(json.loads(dest.read_text()))
            return
        result = {'source_record_id': row['record_id'], 'source_passage_sha256': row['passage_sha256'],
                  'category': row['category'], 'source_id': row.get('source_id'),
                  'parent_document_id': row.get('parent_document_id'),
                  'document_family_id': row.get('document_family_id', row.get('provisional_family_id')),
                  'split': 'development_pilot' if args.pilot else row['split'],
                  'development_exposed': args.pilot, 'model': GENERATOR['model'],
                  'inference_dtype': GENERATOR['inference_dtype'],
                  'training_eligible': False, 'admission_status': 'quarantined_synthetic_candidate',
                  'known_process': 'independent_ai_generation'}
        topic_call = await client.call(request(topic_messages(row), 'topic'), row['record_id'], 'topic')
        result['topic_call_id'] = topic_call['call_id']
        try:
            raw_topic, finish, _ = decode(topic_call['response'])
            if finish != 'stop':
                raise ValueError('Topic generation incomplete')
            topic = parse_topic(raw_topic, row['text'])
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            result['qc'] = {'passed': False, 'flags': ['invalid_topic'], 'detail': str(exc)}
        else:
            writing = await client.call(request(writer_messages(row, topic), 'writer', len(row['text'].split())), row['record_id'], 'writer')
            text, finish, usage = decode(writing['response'])
            qc = assess(row, text, {**usage, 'output_tokens': usage['output_tokens'] or 0}, finish)
            if usage['output_tokens'] is None:
                qc['flags'].append('visible_token_count_unverified')
                qc['passed'] = False
            result.update(topic=topic, text=text, text_sha256=sha(text), writer_call_id=writing['call_id'], usage=usage, qc=qc)
        result['mirror_id'] = sha(row['record_id'] + canonical(identity))
        save(dest, result)
        records.append(result)
        save(args.out / 'status.json', {'state': 'running', 'completed': len(records), 'generated_documents': sum(bool(r.get('text')) for r in records), 'planned': limits['max_documents'], 'cost_usd': client.spent - client.uncertain_cost_reserve, 'uncertain_cost_reserve_usd': client.uncertain_cost_reserve, 'reserved_usd': client.inflight, 'budget_usd': args.budget_usd, 'source_attempt_counts': dict(gate.sources), 'updated_at': datetime.now(timezone.utc).isoformat()})
    queue = iter(rows)
    async def worker():
        nonlocal budget_reached, stop
        while not stop:
            if (args.out / 'stop-requested.json').exists():
                stop = True
                return
            row = next(queue, None)
            if row is None:
                return
            if not gate.claim(row):
                continue
            try:
                await process(row)
            except BudgetExhausted:
                budget_reached = True
                stop = True
            except Exception as exc:
                stop = True
                errors.append({'source_record_id': row['record_id'], 'error': type(exc).__name__ + ': ' + str(exc)})
    await asyncio.gather(*(worker() for _ in range(getattr(args, 'concurrency', 1))))
    with (args.out / 'mirrors.jsonl').open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    if _DURABLE_STORE is not None:
        await asyncio.to_thread(_DURABLE_STORE.persist, args.out / 'mirrors.jsonl')
    state = ('needs_reconciliation' if errors else 'budget_exhausted' if budget_reached else
             'stopped_by_request' if (args.out / 'stop-requested.json').exists() else
             'document_cap_reached' if len(gate.claimed) >= limits['max_documents'] else
             'waiting_for_balanced_sources' if limits.get('source_quotas') else 'complete')
    summary = {'state': state, 'errors': errors, 'completed': len(records), 'generated_documents': sum(bool(r.get('text')) for r in records), 'mechanically_passed': sum(r['qc']['passed'] for r in records),
               'admitted': 0, 'cost_usd': client.spent - client.uncertain_cost_reserve,
               'uncertain_cost_reserve_usd': client.uncertain_cost_reserve, 'budget_accounted_usd': client.spent,
               'source_attempt_counts': dict(gate.sources), 'document_cap': limits['max_documents'],
               'budget_usd': args.budget_usd, 'model': GENERATOR['model']}
    save(args.out / 'status.json', summary)
    print(json.dumps(summary))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--budget-usd', type=float)
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=4)
    asyncio.run(run(parser.parse_args()))
