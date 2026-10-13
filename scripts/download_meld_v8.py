"""Download and checksum-verify the reviewed, pinned MELD v8 release."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "anon-review-meld-2026/meld"
REV = "8990324abd92e1fa17072f6887ea1e5c1cef5abc"
if __name__ == "__main__":
    target = ROOT / "models/meld-v8"
    snapshot_download(
        REPO,
        revision=REV,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "README.md", "meld.py"],
        max_workers=3,
    )
    files = {}
    for entry in HfApi().model_info(REPO, revision=REV, files_metadata=True).siblings:
        path = target / entry.rfilename
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if entry.lfs:
            assert digest == entry.lfs.sha256, entry.rfilename
        files[entry.rfilename] = {"bytes": path.stat().st_size, "sha256": digest}
    assert json.loads((target / "meld_config.json").read_text())["version"] == "v8"
    (target / "download-manifest.json").write_text(
        json.dumps({"repo_id": REPO, "revision": REV, "files": files}, indent=2)
    )
    print(
        "Verified MELD v8:",
        sum(v["bytes"] for v in files.values()),
        "bytes",
        flush=True,
    )
