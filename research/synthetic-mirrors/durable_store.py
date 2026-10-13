"""Direct private-bucket checkpoints, bypassing a failed filesystem mount."""
import hashlib
import os
import time
from pathlib import Path
from urllib.parse import quote


class BucketStore:
    def __init__(self, root, bucket, prefix, api=None, reader=None):
        self.root = Path(root).resolve()
        self.bucket = bucket
        self.prefix = prefix.strip('/')
        if not self.prefix or '..' in self.prefix.split('/'):
            raise ValueError('A scoped bucket prefix is required')
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ['HF_TOKEN'])
        self.api = api
        if reader is None:
            import requests
            def reader(key):
                url = 'https://huggingface.co/buckets/' + self.bucket + '/resolve/' + quote(key, safe='')
                response = requests.get(url, headers={'Authorization': 'Bearer ' + os.environ['HF_TOKEN']}, timeout=60)
                response.raise_for_status()
                return response.content
        self.reader = reader

    @staticmethod
    def _retry(operation):
        # These operations are idempotent uploads/reads of identical bytes, never
        # model requests. A readback mismatch is deliberately not retried.
        import requests
        import httpx
        for attempt in range(4):
            try:
                return operation()
            except (requests.RequestException, httpx.HTTPError) as exc:
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                if attempt == 3 or (status is not None and status not in (408, 429, 500, 502, 503, 504)):
                    raise
                time.sleep(2 ** attempt)

    def _write_verified(self, key, data, digest, kind):
        self._retry(lambda: self.api.batch_bucket_files(self.bucket, add=[(data, key)]))
        actual = self._retry(lambda: self.reader(key))
        if hashlib.sha256(actual).hexdigest() != digest:
            raise RuntimeError('Durable checkpoint ' + kind + ' readback mismatch')

    def persist(self, path):
        path = Path(path).resolve()
        relative = path.relative_to(self.root).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        # Every saved version survives later overwrites of the current key.
        version_key = self.prefix + '/checkpoint-history/' + relative + '/' + digest
        self._write_verified(version_key, data, digest, 'version')
        key = self.prefix + '/' + relative
        self._write_verified(key, data, digest, 'current')


def configured_store():
    names = ['LUNA_CHECKPOINT_ROOT', 'LUNA_BUCKET_ID', 'LUNA_BUCKET_PREFIX']
    values = [os.environ.get(name) for name in names]
    if not any(values):
        return None
    if not all(values):
        raise RuntimeError('Incomplete durable checkpoint configuration')
    return BucketStore(*values)
