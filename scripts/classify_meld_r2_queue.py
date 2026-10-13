"""Stream verified extraction objects from R2 into the optimized CUDA classifier.

Only scoped read access is sent to the GPU. Outputs are atomic local PGF files;
retrieve them and use local account authorization to upload after a rental.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

os.environ.setdefault('RAYON_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from open_text_detector_backend.result_codec import decode, encode


def fetch(access, key, sha):
    req = urllib.request.Request(access['url'] + '/' + key, headers={
        'Authorization': 'Bearer ' + access['token'], 'User-Agent': 'open-text-detector-model-transfer/1.0'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('R2 object checksum mismatch')
            return raw
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-key', required=True)
    p.add_argument('--queue-sha256', required=True)
    p.add_argument('--access-file', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--stop-at', type=float)
    p.add_argument('--verify-only', action='store_true', help='Download and verify the first extraction without requiring CUDA')
    args = p.parse_args()
    access = json.loads(args.access_file.read_text())
    queue = json.loads(fetch(access, args.queue_key, args.queue_sha256))
    assert queue['format'] == 'meld-r2-queue-v1' and queue['model'] == 'meld-v5'
    profile = ROOT / 'models/meld-v5/cuda-runtime-profile.json'
    profile_sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    assert queue['profile_sha256'] == profile_sha, 'Queue requires a different runtime profile'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = args.output_dir / '.r2-cache'
    cache.mkdir(exist_ok=True)
    locks = {}
    verified_bundles = set()

    def get_item(item):
        if 'bundle_key' in item:
            digest = item['bundle_sha256']
            lock = locks.setdefault(digest, threading.Lock())
            path = cache / (digest + '.tar')
            with lock:
                if digest not in verified_bundles:
                    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raw = fetch(access, item['bundle_key'], digest)
                        temporary = path.with_suffix('.tmp')
                        temporary.write_bytes(raw)
                        temporary.replace(path)
                    verified_bundles.add(digest)
            with tarfile.open(path) as archive:
                member = archive.getmember(item['member'])
                assert member.isfile() and member.size <= 128 * 1024 * 1024
                raw = archive.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == item['blob_sha256']
        else:
            raw = fetch(access, item['r2_key'], item['blob_sha256'])
        obj = decode(raw)
        assert obj['pdf_sha256'] == item['pdf_sha256']
        assert obj['text_sha256'] == item['text_sha256'] == hashlib.sha256(obj['text'].encode()).hexdigest()
        assert obj['text'].strip(), 'Empty extraction must be reviewed'
        return item, obj['text']

    pending = []
    for item in queue['papers']:
        path = args.output_dir / (item['text_sha256'] + '.pgf')
        if path.exists():
            result = decode(path.read_bytes())
            assert result['text_sha256'] == item['text_sha256'] and result['cuda_profile_sha256'] == profile_sha
        else:
            pending.append(item)
    if args.verify_only:
        samples = {}
        for item in pending:
            samples.setdefault('bundle' if 'bundle_key' in item else 'individual', item)
        for storage, sample in samples.items():
            item, text = get_item(sample)
            print(json.dumps({'queue_papers': len(queue['papers']), 'storage': storage,
                              'verified_pdf': item['pdf_sha256'], 'characters': len(text)}))
        return
    if not pending:
        print('All queue outputs already completed and verified')
        return
    import torch
    from open_text_detector_backend.providers.meld_cuda import MeldCudaRunner
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = json.loads(profile.read_text())['settings']
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        next_submit = 0

        def prefetch(limit):
            nonlocal next_submit
            while next_submit < min(len(pending), limit):
                futures[next_submit] = pool.submit(get_item, pending[next_submit])
                next_submit += 1

        prefetch(30)
        runner = MeldCudaRunner(ROOT / 'models/meld-v5', **settings)
        runner.warmup()
        for offset in range(0, len(pending), 10):
            if args.stop_at and time.time() >= args.stop_at:
                break
            group = [futures.pop(i).result() for i in range(offset, min(offset + 10, len(pending)))]
            prefetch(offset + 40)
            results = runner.predict_many([text for _, text in group])
            for (item, _), result in zip(group, results, strict=True):
                result.update(text_sha256=item['text_sha256'], pdf_sha256=item['pdf_sha256'],
                              source_maps=[item], cuda_profile_sha256=profile_sha)
                path = args.output_dir / (item['text_sha256'] + '.pgf')
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(encode(result, level=3))
                temporary.replace(path)
                completed += 1
            print(json.dumps({'completed_this_run': completed, 'pending_at_start': len(pending)}), flush=True)
        for future in futures.values():
            future.cancel()


if __name__ == '__main__':
    main()
