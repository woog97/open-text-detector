"""Download staged model parts with an expiring read-only transfer credential."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

manifest = json.loads(Path(sys.argv[1]).read_text())
access = json.loads(Path(sys.argv[2]).read_text())
target = Path(sys.argv[3])
partsdir = target.parent / ".download-parts"
partsdir.mkdir(parents=True, exist_ok=True)


def download(part):
    path = partsdir / part["sha256"]
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]:
        req = urllib.request.Request(access["url"] + "/" + part["key"],
            headers={"Authorization": "Bearer " + access["token"],
                     "User-Agent": "open-text-detector-model-transfer/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        assert len(data) == part["bytes"] and hashlib.sha256(data).hexdigest() == part["sha256"]
        path.write_bytes(data)
    return path


if manifest.get("metadata"):
    metadata_path = download(manifest["metadata"])
    metadata = json.loads(metadata_path.read_text())
    for name, contents in metadata.items():
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Invalid model metadata filename")
        (target.parent / name).write_text(contents)
    metadata_path.unlink()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    parts = list(pool.map(download, manifest["parts"]))
tmp = target.with_suffix(".partial")
digest = hashlib.sha256()
with tmp.open("wb") as f:
    for path in parts:
        data = path.read_bytes()
        digest.update(data)
        f.write(data)
assert digest.hexdigest() == manifest["sha256"]
tmp.replace(target)
for path in parts:
    path.unlink()
print("Checkpoint downloaded and SHA256 verified", manifest["version"], flush=True)
