"""Resumable, content-addressed R2 staging for large public MELD checkpoints.

64 MiB parts fit the existing Cloudflare REST upload path; every part is verified
by MD5/size and the original checkpoint is verified by its pinned SHA256.
Cloudflare account credentials remain local.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_paper_pdfs import request, BASE, ROOT

PART = 64 * 1024 * 1024
OUT = ROOT / "research/benchmarks/vast-meld"


def stage(version):
    directory = ROOT / f"models/meld-{version}"
    manifest = json.loads((directory / "download-manifest.json").read_text())
    path = directory / "model.safetensors"
    expected = manifest["files"]["model.safetensors"]["sha256"]
    with path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    prefix = f"model-cache/meld/{expected}/"
    remote = {x["key"]: x for x in request(BASE + "?" + urllib.parse.urlencode(
        {"prefix": prefix, "per_page": 1000}))["result"]}
    def upload(index):
        with path.open("rb") as f:
            f.seek(index * PART)
            data = f.read(PART)
        key = prefix + f"part-{index:04d}"
        md5 = hashlib.md5(data).hexdigest()
        old = remote.get(key, {})
        if old.get("size") != len(data) or old.get("etag", "").strip('"') != md5:
            request(BASE + "/" + key, data, "application/octet-stream")
        print(f"{version} part {index + 1} ready", flush=True)
        return {"key": key, "bytes": len(data), "md5": md5,
                "sha256": hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(upload, range((path.stat().st_size + PART - 1) // PART)))
    listed = {x["key"]: x for x in request(BASE + "?" + urllib.parse.urlencode(
        {"prefix": prefix, "per_page": 1000}))["result"]}
    for part in parts:
        item = listed[part["key"]]
        assert item["size"] == part["bytes"] and item["etag"].strip('"') == part["md5"]
    result = {"version": version, "revision": manifest["revision"], "sha256": expected,
              "bytes": path.stat().st_size, "parts": parts}
    names = [name for name in manifest["files"] if name.endswith(".json") and Path(name).name == name]
    metadata = {name: (directory / name).read_text() for name in names}
    metadata["download-manifest.json"] = (directory / "download-manifest.json").read_text()
    metadata_bytes = json.dumps(metadata).encode()
    metadata_key = prefix + "part-9999"
    request(BASE + "/" + metadata_key, metadata_bytes, "application/json")
    result["metadata"] = {"key": metadata_key, "bytes": len(metadata_bytes),
                          "sha256": hashlib.sha256(metadata_bytes).hexdigest()}
    raw = json.dumps(result).encode()
    request(BASE + "/" + prefix + "manifest.json", raw, "application/json")
    (OUT / f"r2-{version}-manifest.json").write_bytes(raw)
    print(f"{version} staging verified: {len(parts)} parts, {path.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    for version in sys.argv[1:] or ["v5", "v8"]:
        stage(version)
