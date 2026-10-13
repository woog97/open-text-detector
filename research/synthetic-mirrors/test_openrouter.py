import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mirror_core import sha
from run_openrouter import BalanceGate, BudgetExhausted, Client, GENERATOR, decode, request, reservation, run, save


class HostedTests(unittest.TestCase):
    def test_balance_counts_existing_work_and_reserves_missing_sources(self):
        rows = [{'record_id': str(i), 'source_id': s} for i, s in enumerate(['a', 'a', 'a', 'b', 'b'])]
        gate = BalanceGate(rows, ['0'], {'max_documents': 3, 'source_quotas': {'a': 1, 'b': 1, 'missing': 1}})
        self.assertTrue(gate.claim(rows[0]))
        self.assertFalse(gate.claim(rows[1]))
        self.assertTrue(gate.claim(rows[3]))
        self.assertFalse(gate.claim(rows[4]))
        self.assertEqual(len(gate.claimed), 2)

    def test_document_limit_includes_historical_attempts(self):
        rows = [{'record_id': str(i), 'source_id': 'a'} for i in range(10)]
        gate = BalanceGate(rows, ['0', '1'], {'max_documents': 3})
        self.assertTrue(gate.claim(rows[2]))
        self.assertFalse(gate.claim(rows[3]))
        self.assertEqual(len(gate.claimed), 3)

    def test_interrupted_call_keeps_budget_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save(root / 'calls/interrupted.json', {'state': 'interrupted_reserved_no_replay', 'reserved_usd': .2})
            client = Client(root, 1, None, GENERATOR['model'])
            self.assertEqual(client.spent, .2)
            self.assertEqual(client.uncertain_cost_reserve, .2)

    def test_concurrent_requests_share_one_budget(self):
        async def scenario(root):
            body = request([], 'topic')
            cost = reservation(body)
            dispatched = []
            async def fetch(*args, **kwargs):
                dispatched.append(1)
                await asyncio.sleep(.01)
                return 200, {'model': GENERATOR['model'], 'service_tier': 'flex', 'usage': {'cost': cost}}
            client = Client(root, cost * 2.1, fetch, GENERATOR['model'])
            results = await asyncio.gather(*(client.call(body, str(i), 'topic') for i in range(8)), return_exceptions=True)
            self.assertEqual(len(dispatched), 2)
            self.assertEqual(sum(isinstance(r, BudgetExhausted) for r in results), 6)
            self.assertLessEqual(client.spent, cost * 2.1)
            self.assertAlmostEqual(client.inflight, 0)
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(scenario(Path(folder)))

    def test_hidden_reasoning_is_not_visible_text(self):
        raw = {'choices': [{'message': {'content': 'passage'}, 'finish_reason': 'stop'}],
               'usage': {'prompt_tokens': 40, 'completion_tokens': 200,
                         'completion_tokens_details': {'reasoning_tokens': 150}}}
        self.assertEqual(decode(raw)[2]['output_tokens'], 50)
        del raw['usage']['completion_tokens_details']
        self.assertIsNone(decode(raw)[2]['output_tokens'])

    def test_budget_stops_before_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            async def never(*args, **kwargs):
                self.fail('Over-budget request was dispatched')
            client = Client(Path(folder), .000001, never, GENERATOR['model'])
            with self.assertRaisesRegex(RuntimeError, 'Budget exhausted'):
                asyncio.run(client.call(request([], 'topic'), 'parent', 'topic'))
            self.assertFalse((Path(folder) / 'calls').exists())

    def test_uncertain_dispatch_blocks_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder) / 'calls/unknown.json', {'state': 'dispatched'})
            with self.assertRaisesRegex(RuntimeError, 'Unresolved dispatch'):
                Client(Path(folder), 1, None, GENERATOR['model'])

    def test_complete_pipeline_resumes_without_paid_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = '\u2028'.join('sourceword' + str(i) for i in range(100))
            source = {'record_id': 'parent/unsafe-filename', 'text': text, 'category': 'creative',
                      'passage_sha256': sha(text), 'source_id': 'fixture', 'provisional_family_id': 'family'}
            (root / 'input.jsonl').write_text(json.dumps(source, ensure_ascii=False) + '\n')
            calls = []
            async def fetch(path, body=None, auth=True):
                if path.endswith('/endpoints'):
                    return 200, {'data': {'endpoints': [{'tag': 'openai/flex', 'pricing': {
                        'prompt': '0.00000005', 'completion': '0.00000025', 'input_cache_write': '0.0000000625'}}]}}
                if path == 'models':
                    return 200, {'data': [{'id': GENERATOR['model'], 'canonical_slug': 'openai/gpt-6-luna-20260922'}]}
                calls.append(body)
                topic = 'response_format' in body
                output = json.dumps({'topic': 'A traveler finding a new home'}) if topic else ' '.join('newword' + str(i) for i in range(100))
                return 200, {'model': GENERATOR['model'], 'service_tier': 'flex',
                             'choices': [{'message': {'content': output}, 'finish_reason': 'stop'}],
                             'usage': {'cost': .00001, 'prompt_tokens': 20, 'completion_tokens': 200,
                                       'completion_tokens_details': {'reasoning_tokens': 50}}}
            args = SimpleNamespace(input=root / 'input.jsonl', out=root / 'out', limit=1, pilot=True, execute=True, budget_usd=1)
            asyncio.run(run(args, fetch))
            asyncio.run(run(args, fetch))
            self.assertEqual(len(calls), 2)
            self.assertNotIn(text, json.dumps(calls[1]))
            row = json.loads((args.out / 'mirrors.jsonl').read_text())
            self.assertTrue(row['qc']['passed'])
            self.assertFalse(row['training_eligible'])
            self.assertEqual(row['document_family_id'], 'family')
            self.assertEqual(row['split'], 'development_pilot')
            self.assertEqual(row['inference_dtype'], 'provider_managed_unreported')
            self.assertEqual(json.loads((args.out / 'status.json').read_text())['cost_usd'], .00002)

    def test_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            count = []
            async def fail(*args, **kwargs):
                count.append(1)
                return 0, {'error': {'type': 'transport'}}
            client = Client(Path(folder), 1, fail, GENERATOR['model'])
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    asyncio.run(client.call(request([], 'topic'), 'parent', 'topic'))
            self.assertEqual(len(count), 1)

if __name__ == '__main__':
    unittest.main()
