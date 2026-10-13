"""Acquire the pinned gated source with an authorized Hugging Face read token."""

import json

import httpx

from arena20 import B, D, sha

REV = "1b6335d42a1d2c7e34870c905d03ab964f7f2bd8"
FILE = "data/train-00000-of-00001-cced8514c7ed782a.parquet"
EXPECTED = "3726a6352e9bfc34e206460646f6e5e99bb837751966a671ddd30c7f64e5b06e"
URL = f"https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/resolve/{REV}/{FILE}"


def main():
    dest = D / "source.parquet"
    if dest.exists():
        assert sha(dest.read_bytes()) == EXPECTED
        print("Pinned source already present and checksum verified")
        return
    token = next(
        (
            line.split("=", 1)[1].strip().strip("\"'")
            for line in (B / ".env.secrets").read_text().split("\n")
            if line.startswith("HF_TOKEN=")
        ),
        None,
    )
    if not token:
        raise ValueError(
            "Add an authorized read token to HF_TOKEN in .env.secrets; accept dataset access conditions in your own account first."
        )
    # httpx removes the Authorization header on redirects to a different origin.
    response = httpx.get(
        URL,
        headers={"Authorization": "Bearer " + token},
        follow_redirects=True,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Dataset access failed: HTTP {response.status_code}; no credential is printed"
        )
    assert sha(response.content) == EXPECTED
    D.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    (D / "source_manifest.json").write_text(
        json.dumps(
            {
                "url": URL,
                "revision": REV,
                "bytes": len(response.content),
                "sha256": EXPECTED,
            },
            indent=2,
        )
        + "\n"
    )
    print("Downloaded verified source bytes:", len(response.content))


if __name__ == "__main__":
    main()
