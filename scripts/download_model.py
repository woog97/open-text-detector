"""Download and checksum the reviewed merged model; run with backend/.venv/bin/python."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", str(ROOT / "models/.hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from open_text_detector_backend.providers.checkpoints import QWEN_ID, QWEN_REVISION  # noqa: E402
from huggingface_hub import snapshot_download, HfApi  # noqa: E402


def main():
    target = ROOT / "models/editlens-qwen3-4b-merged-v3"
    snapshot_download(QWEN_ID, revision=QWEN_REVISION, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.md", "*.jinja"], max_workers=3)
    info = HfApi().model_info(QWEN_ID, revision=QWEN_REVISION, files_metadata=True)
    files = {}
    for entry in info.siblings:
        path = target / entry.rfilename
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if entry.lfs and digest != entry.lfs.sha256:
            raise ValueError(f"Checksum mismatch: {entry.rfilename}")
        files[entry.rfilename] = {"bytes": path.stat().st_size, "sha256": digest}
    manifest = {"repo_id": QWEN_ID, "revision": QWEN_REVISION, "files": files,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "license": "CC-BY-NC-SA-4.0"}
    (target / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model": QWEN_ID, "revision": QWEN_REVISION,
                      "verified_bytes": sum(f["bytes"] for f in files.values())}))


if __name__ == "__main__":
    main()
