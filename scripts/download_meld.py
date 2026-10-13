"""Install the reviewed MELD v5 checkpoint, pinned and checksum verified."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", str(ROOT / "models/.hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from open_text_detector_backend.providers.checkpoints import MELD_ID, MELD_REVISION  # noqa: E402
from huggingface_hub import snapshot_download, HfApi  # noqa: E402


def main():
    target = ROOT / "models/meld-v5"
    snapshot_download(MELD_ID, revision=MELD_REVISION, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "README.md"], max_workers=3)
    info = HfApi().model_info(MELD_ID, revision=MELD_REVISION, files_metadata=True)
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
    manifest = {"repo_id": MELD_ID, "revision": MELD_REVISION, "files": files,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "license": "MIT"}
    (target / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model": MELD_ID, "revision": MELD_REVISION,
                      "verified_bytes": sum(f["bytes"] for f in files.values())}))


if __name__ == "__main__":
    main()
