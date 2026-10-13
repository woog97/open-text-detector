"""Upload only the reviewed dataset package, never the benchmark work directory."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

from arena20 import B

PACKAGE = B / "exports/arena-prose-100-49-models"
REPO_ID = "woog/arena-prose-100-49-models"
FILES = [
    "README.md",
    "LICENSE.md",
    "models.json",
    "provenance.json",
    "data/test-00000-of-00001.parquet",
]


def main():
    token = next(
        s.split("=", 1)[1].strip().strip('"\x27')
        for s in (B / ".env.secrets").read_text().splitlines()
        if s.startswith("HF_TOKEN=")
    )
    api = HfApi(token=token)
    assert api.whoami()["name"] == REPO_ID.split("/")[0]
    assert sorted(
        p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()
    ) == sorted(FILES)
    assert "REPO_ID_PLACEHOLDER" not in (PACKAGE / "README.md").read_text()
    try:
        existing = api.repo_info(REPO_ID, repo_type="dataset")
    except RepositoryNotFoundError:
        existing = None
    if existing is not None:
        # Never overwrite a pre-existing remote dataset without a local upload receipt.
        receipt = B / "runs/arena100-expanded/huggingface_upload.json"
        if not receipt.exists():
            assert set(api.list_repo_files(REPO_ID, repo_type="dataset")) <= {
                ".gitattributes"
            }, "Refusing to overwrite existing data"
        else:
            assert json.loads(receipt.read_text())["repo_id"] == REPO_ID
        assert not existing.private
    else:
        api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=False)
    old_receipt = B / "runs/arena100-expanded/huggingface_upload.json"
    if old_receipt.exists():
        saved = json.loads(old_receipt.read_text())
        archive = old_receipt.with_name(
            "huggingface_upload_" + saved["commit"] + ".json"
        )
        if not archive.exists():
            archive.write_bytes(old_receipt.read_bytes())
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(PACKAGE),
        allow_patterns=FILES,
        commit_message="Add 100 Kimi K3 responses; expand to 50 models and 5,000 rows",
        parent_commit=existing.sha if existing is not None else None,
    )
    receipt = {
        "repo_id": REPO_ID,
        "url": f"https://huggingface.co/datasets/{REPO_ID}",
        "commit": commit.oid,
        "public": True,
        "verified_files": {},
    }
    receipt_path = B / "runs/arena100-expanded/huggingface_upload.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    remote = api.repo_info(REPO_ID, repo_type="dataset", revision=commit.oid)
    assert not remote.private
    names = set(api.list_repo_files(REPO_ID, repo_type="dataset", revision=commit.oid))
    assert set(FILES) <= names
    assert names <= set(FILES) | {".gitattributes"}
    for name in FILES:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=name,
            revision=commit.oid,
            token=token,
            cache_dir="/tmp/arena-hf-upload-verify",
        )
        data = Path(downloaded).read_bytes()
        assert data == (PACKAGE / name).read_bytes(), f"Remote content mismatch: {name}"
        receipt["verified_files"][name] = hashlib.sha256(data).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc).replace(
                        next(
                            s.split("=", 1)[1].strip().strip('"\x27')
                            for s in (B / ".env.secrets").read_text().splitlines()
                            if s.startswith("HF_TOKEN=")
                        ),
                        "[REDACTED]",
                    ),
                    "status": getattr(
                        getattr(exc, "response", None), "status_code", None
                    ),
                }
            )
        )
        raise SystemExit(1)
