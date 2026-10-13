import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from durable_store import BucketStore
from run_openrouter import Client, GENERATOR, request, save


class MemoryBucket:
    def __init__(self):
        self.objects = {}

    def batch_bucket_files(self, bucket, add):
        for data, key in add:
            self.objects[key] = data


class DurabilityTests(unittest.TestCase):
    def test_transient_checkpoint_failure_retries_bytes_not_paid_call(self):
        import requests
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);api=MemoryBucket();failures=[2];paid=[]
            def reader(key):
                if failures[0]:
                    failures[0]-=1;raise requests.ConnectionError('temporary disconnect')
                return api.objects[key]
            store=BucketStore(root,'b','p',api,reader)
            async def fetch(*args):
                paid.append(1)
                return 200,{'model':GENERATOR['model'],'service_tier':GENERATOR['service_tier'],'usage':{'cost':0.00001}}
            client=Client(root,20,fetch,GENERATOR['model'])
            with patch('run_openrouter._DURABLE_STORE',store),patch('durable_store.time.sleep'):
                result=asyncio.run(client.call(request([], 'topic'),'parent','topic'))
            self.assertEqual(result['state'],'complete');self.assertEqual(len(paid),1)

    def test_versions_and_current_payload_survive_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); api = MemoryBucket()
            store = BucketStore(root, 'private/bucket', 'run', api, api.objects.__getitem__)
            path = root/'calls/call.json'; path.parent.mkdir()
            with patch('run_openrouter._DURABLE_STORE', store):
                save(path, {'state':'dispatched'})
                save(path, {'state':'complete','cost_usd':.01})
            versions = [json.loads(v) for k,v in api.objects.items() if '/checkpoint-history/' in k]
            self.assertEqual({v['state'] for v in versions}, {'dispatched','complete'})
            self.assertEqual(json.loads(api.objects['run/calls/call.json'])['state'], 'complete')

    def test_failed_remote_readback_prevents_any_paid_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);api=MemoryBucket()
            store=BucketStore(root,'private/bucket','run',api,lambda key:b'corrupt')
            async def never(*args):
                self.fail('Paid call dispatched before durable checkpoint verification')
            client=Client(root,20,never,GENERATOR['model'])
            with patch('run_openrouter._DURABLE_STORE',store):
                with self.assertRaisesRegex(RuntimeError,'readback mismatch'):
                    asyncio.run(client.call(request([], 'topic'),'parent','topic'))
            self.assertTrue(client.halted)
            self.assertTrue(list((root/'calls').glob('*.json')))

    def test_no_checkpoints_outside_configured_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);api=MemoryBucket();store=BucketStore(root/'run','b','p',api,api.objects.__getitem__)
            outside=root/'outside';outside.write_text('private')
            with self.assertRaises(ValueError):store.persist(outside)
            self.assertEqual(api.objects,{})


if __name__=='__main__':unittest.main()
