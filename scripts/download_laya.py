"""Download only the pinned English Laya checkpoint; verify Hub hashes."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
os.environ.setdefault('HF_HOME', str(ROOT / 'models/.hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import HfApi, snapshot_download
from open_text_detector_backend.providers.checkpoints import LAYA_ID, LAYA_REVISION


def main():
    target = ROOT / 'models/laya'
    names = ['model.safetensors', 'rl_agent_config.json', 'encoder/config.json',
             'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'README.md']
    snapshot_download(LAYA_ID, revision=LAYA_REVISION, local_dir=target, allow_patterns=names)
    info = HfApi().model_info(LAYA_ID, revision=LAYA_REVISION, files_metadata=True)
    entries = {e.rfilename: e for e in info.siblings}
    files = {}
    for name in names:
        path = target / name
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        e = entries[name]
        if e.lfs:
            if digest != e.lfs.sha256:
                raise ValueError('Checksum mismatch: ' + name)
        else:
            if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != e.blob_id:
                raise ValueError('Git blob checksum mismatch: ' + name)
        files[name] = {'sha256': digest, 'bytes': len(data)}
    (target / 'download-manifest.json').write_text(json.dumps(
        {'repo_id': LAYA_ID, 'revision': LAYA_REVISION, 'license': 'Apache-2.0', 'files': files}, indent=2))
    print(json.dumps({'revision': LAYA_REVISION, 'verified_bytes': sum(f['bytes'] for f in files.values())}))


if __name__ == '__main__':
    main()
